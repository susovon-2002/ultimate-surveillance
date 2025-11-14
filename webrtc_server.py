## webrtc_full.py
"""
Single-file WebSocket JPEG streaming YOLO server with:
- Multi-user / multi-camera support
- FPS graphs, detection log (CSV), screenshots
- Optional ONNX face recognition
- Stream recording (MP4)
- Admin panel (ENV ADMIN_PW)
- ADDED: object name + count display, FIRE alarm (visual + sound)
"""

import os, base64, time, threading, csv, io
from datetime import datetime
from collections import defaultdict, deque

from flask import Flask, render_template_string, request, jsonify, send_file, abort
from flask_socketio import SocketIO, emit

import cv2, numpy as np

# ---------- CONFIG ----------
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.30"))
PREVIEW_WIDTH = int(os.environ.get("PREVIEW_WIDTH", "640"))
DEVICE = os.environ.get("DEVICE", "cpu")
ADMIN_PW = os.environ.get("ADMIN_PW", "adminpassword")
OUT_DIR = os.environ.get("OUT_DIR", "ultimate_detections")
os.makedirs(OUT_DIR, exist_ok=True)
SNAP_DIR = os.path.join(OUT_DIR, "snapshots"); os.makedirs(SNAP_DIR, exist_ok=True)
LOG_CSV = os.path.join(OUT_DIR, "detections_log.csv")
ATT_CSV = os.path.join(OUT_DIR, "attendance.csv")
RECORD_DIR = os.path.join(OUT_DIR, "recordings"); os.makedirs(RECORD_DIR, exist_ok=True)

# Optional ONNX face models env paths
FACE_DETECTOR_ONNX = os.environ.get("FACE_DETECTOR_ONNX", "models/ultraface_slim.onnx")
FACE_RECOG_ONNX = os.environ.get("FACE_RECOG_ONNX", "models/mobilefacenet_arcface.onnx")
FACE_DB_DIR = os.environ.get("FACE_DB_DIR", "faces"); os.makedirs(FACE_DB_DIR, exist_ok=True)
FACE_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "0.45"))

# timeline config
TIMELINE_LENGTH = 60
TIMELINE_INTERVAL = 0.5

# ---------- GLOBAL STATE ----------
clients = {}   # sid -> client info dict
clients_lock = threading.Lock()
stats = {}     # sid -> {"fps":..., "last_ts":..., "proc_ms":...}
stats_lock = threading.Lock()

detection_log_lock = threading.Lock()

# recording: sid -> {"writer": VideoWriter, "path": str, "width":int, "height":int}
recordings = {}
recording_lock = threading.Lock()

# Object counts per sid
object_counts = defaultdict(lambda: defaultdict(int))
object_counts_lock = threading.Lock()

# timeline per-sid
timelines = defaultdict(lambda: deque([0]*TIMELINE_LENGTH, maxlen=TIMELINE_LENGTH))

# face ONNX
try:
    import onnxruntime as ort
    ONNXRT = True
except Exception:
    ONNXRT = False

# YOLO
try:
    from ultralytics import YOLO
    ymodel = YOLO(YOLO_MODEL)
    print("[INFO] YOLO loaded:", YOLO_MODEL)
except Exception as e:
    print("[WARN] YOLO not available:", e)
    ymodel = None

ymodel_lock = threading.Lock()

# face models (optional)
face_session = None
face_rec_session = None
face_detector_available = False
face_rec_available = False

def load_face_models():
    global face_session, face_rec_session, face_detector_available, face_rec_available
    if not ONNXRT:
        return
    try:
        if os.path.exists(FACE_DETECTOR_ONNX):
            face_session = ort.InferenceSession(FACE_DETECTOR_ONNX, providers=["CPUExecutionProvider"])
            face_detector_available = True
        if os.path.exists(FACE_RECOG_ONNX):
            face_rec_session = ort.InferenceSession(FACE_RECOG_ONNX, providers=["CPUExecutionProvider"])
            face_rec_available = True
    except Exception as e:
        print("Face model load error:", e)

if ONNXRT:
    load_face_models()

def l2_norm(x):
    x = x.astype(np.float32)
    x /= np.linalg.norm(x) + 1e-6
    return x

def get_face_embedding(img):
    if not face_rec_available:
        return None
    try:
        inp = cv2.resize(img, (112,112)).astype(np.float32)/255.0
        inp = np.transpose(inp, (2,0,1))[None,:,:,:].astype(np.float32)
        out = face_rec_session.run(None, {face_rec_session.get_inputs()[0].name: inp})
        emb = np.array(out[0][0])
        return l2_norm(emb)
    except Exception:
        return None

def find_match(emb):
    best = None; best_name = None
    for fn in os.listdir(FACE_DB_DIR):
        if not fn.lower().endswith(".npy"): continue
        name = os.path.splitext(fn)[0]
        try:
            db_emb = np.load(os.path.join(FACE_DB_DIR, fn))
            dist = np.dot(db_emb, emb)
            if best is None or dist > best:
                best = dist; best_name = name
        except Exception:
            pass
    return best_name, float(best) if best is not None else (None, None)

def save_face_sample(name, crop):
    emb = get_face_embedding(crop)
    if emb is None:
        return False
    path = os.path.join(FACE_DB_DIR, f"{name}.npy")
    np.save(path, emb)
    append_attendance(name)
    return True

# ---------- Logging helpers ----------
def append_detection_log(row):
    header = ["timestamp","sid","camera_name","class","conf","image_path"]
    with detection_log_lock:
        exists = os.path.exists(LOG_CSV)
        try:
            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(header)
                w.writerow(row)
        except Exception as e:
            print("Log write error:", e)

def append_attendance(name):
    exists = os.path.exists(ATT_CSV)
    try:
        with open(ATT_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["timestamp","name"])
            w.writerow([datetime.now().isoformat(), name])
    except Exception as e:
        print("Attendance write error:", e)

# ---------- Helpers ----------
def timestamped_filename(prefix, sid, ext="jpg"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SNAP_DIR, f"{prefix}_{sid}_{ts}.{ext}")

def decode_b64_image(b64):
    try:
        if "," in b64:
            b64 = b64.split(",",1)[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def encode_b64_image(img, quality=80):
    try:
        ret, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii")
    except Exception:
        return None

# ---------- Annotate + count + log + fire alert ----------
# Define alert classes (fire alarm will trigger on 'fire' label)
ALERT_CLASSES = set(["fire", "gun", "knife", "weapon", "person", "vehicle"])
ALERT_COOLDOWN = 60  # seconds cooldown per (sid,label)
last_alert_ts = {}
alerts_lock = threading.Lock()

def emit_fire_alert(sid, camera_name):
    payload = {"sid": sid, "camera": camera_name, "time": datetime.now().isoformat()}
    try:
        socketio.emit("fire_alert", payload)
    except Exception as e:
        print("Fire alert emit error:", e)

def annotate_frame(frame, results, sid, camera_name):
    ann = frame.copy()
    try:
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None: continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD: continue
                cls = int(box.cls[0])
                name = r.names.get(cls, str(cls)).lower()
                xy = box.xyxy[0].cpu().numpy().astype(int) if hasattr(box.xyxy[0], "cpu") else box.xyxy[0].numpy().astype(int)
                x1,y1,x2,y2 = [int(v) for v in xy]
                color = (0,0,255)
                cv2.rectangle(ann, (x1,y1), (x2,y2), color, 2)
                label = f"{name} {conf:.2f}"
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                by1 = max(0, y1 - th - 6)
                cv2.rectangle(ann, (x1, by1), (x1 + tw + 6, y1), color, -1)
                cv2.putText(ann, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Increment object count for this sid/name
                try:
                    with object_counts_lock:
                        object_counts[sid][name] += 1
                        count_val = object_counts[sid][name]
                    # show count under the box
                    cv2.putText(ann, f"Count: {count_val}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                except Exception:
                    count_val = 0

                # save detection snapshot and append log
                try:
                    snap = frame[max(0,y1-10):min(frame.shape[0], y2+10), max(0,x1-10):min(frame.shape[1], x2+10)]
                    path = timestamped_filename("det", sid)
                    cv2.imwrite(path, snap)
                except Exception:
                    path = ""
                append_detection_log([datetime.now().isoformat(), sid, camera_name, name, f"{conf:.3f}", path])

                # Fire-specific behavior: emit fire_alert and update UI
                if name == "fire":
                    # rate-limit fire alerts per sid
                    key = (sid, "fire")
                    with alerts_lock:
                        tprev = last_alert_ts.get(key, 0)
                        nowt = time.time()
                        if nowt - tprev > ALERT_COOLDOWN:
                            last_alert_ts[key] = nowt
                            # emit event to clients
                            try:
                                emit_fire_alert(sid, camera_name)
                            except Exception:
                                pass
    except Exception as e:
        print("Annotate error:", e)
    return ann

def run_yolo_and_faces(frame, sid, camera_name):
    if ymodel is None:
        return frame, 0
    try:
        with ymodel_lock:
            results = ymodel.predict(frame, imgsz=PREVIEW_WIDTH, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
        # annotate (counts & logs inside annotate_frame)
        out = annotate_frame(frame, results, sid, camera_name)
        # optional face recognition as before
        if face_detector_available and face_rec_available:
            try:
                for r in results:
                    boxes = getattr(r, "boxes", None)
                    if boxes is None: continue
                    for box in boxes:
                        cls = int(box.cls[0]); name = r.names.get(cls, str(cls)).lower()
                        if name not in ("person","people"): continue
                        xy = box.xyxy[0].cpu().numpy().astype(int) if hasattr(box.xyxy[0], "cpu") else box.xyxy[0].numpy().astype(int)
                        x1,y1,x2,y2 = [int(v) for v in xy]
                        fh1 = y1; fh2 = min(y2, y1 + (y2 - y1)//2)
                        crop = frame[fh1:fh2, x1:x2]
                        if crop.size == 0: continue
                        emb = get_face_embedding(crop)
                        if emb is not None:
                            match, score = find_match(emb)
                            if match and score > 0.5:
                                cv2.putText(out, f"{match} ({score:.2f})", (x1, y2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                                append_attendance(match)
            except Exception:
                pass
        return out, 0
    except Exception as e:
        print("YOLO/face error:", e)
        return frame, 0

# ---------- Flask + SocketIO ----------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    with clients_lock:
        clients[sid] = {"sid": sid, "camera_name": None, "connected_at": datetime.now().isoformat(), "last_seen": datetime.now().isoformat()}
    print(f"[CONNECT] {sid}")
    emit("connected", {"sid": sid})
    broadcast_clients()

def broadcast_clients():
    with clients_lock:
        simple = [{ "sid": c["sid"], "camera_name": c.get("camera_name"), "last_seen": c.get("last_seen") } for c in clients.values()]
    socketio.emit("clients", simple)

@socketio.on("register")
def handle_register(data):
    sid = request.sid
    camera_name = data.get("camera_name", f"camera_{sid[:6]}")
    with clients_lock:
        if sid in clients:
            clients[sid]["camera_name"] = camera_name
            clients[sid]["last_seen"] = datetime.now().isoformat()
        else:
            clients[sid] = {"sid":sid, "camera_name":camera_name, "connected_at": datetime.now().isoformat(), "last_seen": datetime.now().isoformat()}
    print(f"[REGISTER] {sid} as {camera_name}")
    broadcast_clients()

@socketio.on("frame")
def handle_frame(data):
    sid = request.sid
    now = time.time()
    b64 = data.get("image", "")
    camera_name = data.get("camera_name", clients.get(sid, {}).get("camera_name", f"cam_{sid[:6]}"))
    img = decode_b64_image(b64)
    if img is None:
        return
    # stats fps
    with stats_lock:
        s = stats.setdefault(sid, {"last_ts": now, "fps": 0.0, "proc_ms":0})
        dt = now - s["last_ts"] if s["last_ts"] else 0.001
        s["fps"] = 0.9 * s["fps"] + 0.1 * (1.0/dt if dt>0 else 0)
        s["last_ts"] = now

    # motion detection placeholder can be added here if desired (earlier code had it)
    # run detection (synchronously)
    t0 = time.time()
    out, _ = run_yolo_and_faces(img, sid, camera_name)
    proc_ms = int((time.time() - t0) * 1000)
    with stats_lock:
        stats[sid]["proc_ms"] = proc_ms

    # push value to timeline
    try:
        count_val = 0
        timelines[sid].append(count_val)
    except:
        pass

    # recording if enabled
    with recording_lock:
        rec = recordings.get(sid)
        if rec and rec.get("writer") is not None:
            try:
                rec["writer"].write(out)
            except Exception as e:
                print("Recording write error:", e)

    # encode and emit processed image back to client
    out_b64 = encode_b64_image(out, quality=80)

    # send detection counts to client
    with object_counts_lock:
        counts_copy = dict(object_counts.get(sid, {}))

    emit("processed", {"image": out_b64, "proc_ms": proc_ms})
    emit("det_counts", {"counts": counts_copy})

    # update last_seen & broadcast
    with clients_lock:
        if sid in clients:
            clients[sid]["last_seen"] = datetime.now().isoformat()
    broadcast_clients()

@socketio.on("screenshot")
def handle_screenshot(data):
    sid = request.sid
    b64 = data.get("image", "")
    img = decode_b64_image(b64)
    if img is None:
        emit("screenshot_ack", {"ok": False, "msg": "No image"})
        return
    path = timestamped_filename("screenshot", sid)
    try:
        cv2.imwrite(path, img)
        emit("screenshot_ack", {"ok": True, "path": os.path.basename(path)})
    except Exception as e:
        emit("screenshot_ack", {"ok": False, "msg": str(e)})

@socketio.on("start_record")
def handle_start_record(data):
    sid = request.sid
    fname = f"rec_{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    path = os.path.join(RECORD_DIR, fname)
    width = int(data.get("width", PREVIEW_WIDTH))
    height = int(data.get("height", int(PREVIEW_WIDTH*0.75)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        writer = cv2.VideoWriter(path, fourcc, float(data.get("fps", 10.0)), (width, height))
        with recording_lock:
            recordings[sid] = {"writer": writer, "path": path, "width": width, "height": height}
        emit("record_started", {"ok": True, "path": os.path.basename(path)})
    except Exception as e:
        emit("record_started", {"ok": False, "msg": str(e)})

@socketio.on("stop_record")
def handle_stop_record():
    sid = request.sid
    with recording_lock:
        rec = recordings.pop(sid, None)
    if rec and rec.get("writer") is not None:
        try:
            rec["writer"].release()
            emit("record_stopped", {"ok": True, "path": os.path.basename(rec["path"])})
        except Exception as e:
            emit("record_stopped", {"ok": False, "msg": str(e)})
    else:
        emit("record_stopped", {"ok": False, "msg": "No recording"})

@socketio.on("enroll_face")
def handle_enroll_face(data):
    sid = request.sid
    name = data.get("name","").strip()
    b64 = data.get("image","")
    if not name:
        emit("enroll_ack", {"ok": False, "msg": "Name required"})
        return
    img = decode_b64_image(b64)
    if img is None:
        emit("enroll_ack", {"ok": False, "msg": "No image"})
        return
    # save central crop
    h,w = img.shape[:2]
    cx, cy = w//2, h//3
    wbox = min(200, w//3)
    crop = img[max(0,cy-wbox//2):min(h,cy+wbox//2), max(0,cx-wbox//2):min(w,cx+wbox//2)]
    ok = save_face_sample(name, crop)
    emit("enroll_ack", {"ok": ok})

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"[DISCONNECT] {sid}")
    with clients_lock:
        clients.pop(sid, None)
    with stats_lock:
        stats.pop(sid, None)
    with recording_lock:
        rec = recordings.pop(sid, None)
        if rec and rec.get("writer"):
            try: rec["writer"].release()
            except: pass
    broadcast_clients()

# ---------- HTTP endpoints ----------
@app.route("/")
def index():
    return render_template_string(MAIN_PAGE_HTML, timeline_len=TIMELINE_LENGTH)

@app.route("/admin")
def admin_page():
    return render_template_string(ADMIN_PAGE_HTML)

@app.route("/api/clients")
def api_clients():
    pw = request.args.get("pw","")
    if pw != ADMIN_PW:
        return jsonify({"ok":False, "msg":"auth required"}), 401
    with clients_lock:
        return jsonify(list(clients.values()))

@app.route("/api/stats")
def api_stats():
    with stats_lock:
        return jsonify(stats)

@app.route("/api/timelines")
def api_timelines():
    # per-sid timelines
    out = {}
    for sid, dq in timelines.items():
        out[sid] = list(dq)
    return jsonify(out)

@app.route("/download_log")
def download_log():
    if os.path.exists(LOG_CSV):
        return send_file(LOG_CSV, as_attachment=True)
    return "No log", 404

@app.route("/download_record/<fname>")
def download_record(fname):
    path = os.path.join(RECORD_DIR, fname)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Not found", 404

# Serve snapshots
@app.route("/snapshots/<fname>")
def serve_snapshot(fname):
    path = os.path.join(SNAP_DIR, fname)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "Not found", 404

# ---------- EMBEDDED HTML (client + dashboard + admin) ----------
MAIN_PAGE_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ultimate Multi-Cam (WebSocket YOLO)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<style>
body{background:#0b0b0b;color:#eee;font-family:Arial;margin:0;padding:12px}
header{display:flex;justify-content:space-between;align-items:center}
.container{display:flex;gap:12px}
.left{flex:1}
.right{width:420px;position:sticky;top:12px}
.card{background:#121212;padding:10px;border-radius:8px;margin-bottom:12px}
video,img{width:100%;border-radius:6px;border:2px solid #222}
.controls{display:flex;gap:6px;margin-top:8px}
.small{font-size:0.9em;color:#bbb}
.btn{background:#2b2b2b;border:0;color:#fff;padding:8px;border-radius:6px;cursor:pointer}
.input{padding:6px;border-radius:6px;border:1px solid #333;background:#0b0b0b;color:#eee}
.client-list{max-height:200px;overflow:auto}
.log-list{max-height:200px;overflow:auto}
.alert-banner{background:#b21010;color:#fff;padding:8px;border-radius:6px;text-align:center;font-weight:bold}
</style>
</head>
<body>
<header>
  <h2>Ultimate Multi-Cam — WebSocket YOLO</h2>
  <div><a href="/admin" style="color:#9cf">Admin</a></div>
</header>
<div class="container">
  <div class="left">
    <div class="card">
      <div style="display:flex;gap:8px">
        <div style="flex:1"><input id="camera_name" class="input" placeholder="Camera name (optional)"/></div>
        <button id="start" class="btn">Start</button>
        <button id="stop" class="btn">Stop</button>
      </div>
      <div style="display:flex;gap:12px;margin-top:12px">
        <div style="flex:1">
          <h4>Local</h4>
          <video id="local_video" autoplay playsinline></video>
        </div>
        <div style="width:420px">
          <h4>Processed (Server)</h4>
          <div id="fire_banner" style="display:none" class="alert-banner">🔥 FIRE ALERT</div>
          <img id="proc_img" src="">
          <div class="small" id="proc_info"></div>
        </div>
      </div>

      <div style="display:flex;gap:8px;margin-top:8px">
        <button id="screenshot" class="btn">Screenshot</button>
        <button id="start_rec" class="btn">Start Record</button>
        <button id="stop_rec" class="btn">Stop Record</button>
        <input id="enroll_name" class="input" placeholder="Name to enroll face"/>
        <button id="enroll" class="btn">Enroll</button>
      </div>
    </div>

    <div class="card">
      <h4>FPS Graph</h4>
      <canvas id="fpsChart" height="80"></canvas>
    </div>

    <div class="card">
      <h4>Detections Log (latest)</h4>
      <div id="loglist" class="log-list small">Waiting...</div>
      <div style="margin-top:8px"><a id="download_log" href="/download_log">Download CSV</a></div>
    </div>

  </div>

  <div class="right">
    <div class="card">
      <h4>Connected Clients</h4>
      <div id="clients" class="client-list small">No clients</div>
    </div>

    <div class="card">
      <h4>Recordings</h4>
      <div id="recs" class="small">No recordings</div>
    </div>
  </div>
</div>

<!-- Alarm sound (browser will play when fire_alert is received) -->
<audio id="alarm_sound" src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" preload="auto"></audio>

<script>
let socket = io({transports:["websocket"]});
let localStream = null;
let sending = false;
let fpsSend = 4;
let captureInterval = null;
let sid = null;

socket.on("connect", ()=>{ console.log("connected"); });
socket.on("connected", (d)=>{ sid = d.sid; });

socket.on("clients", (list)=> {
  const el = document.getElementById("clients");
  el.innerHTML = "";
  list.forEach(c=>{
    const div = document.createElement("div");
    div.innerText = `${c.camera_name||"-"}  (${c.sid.slice(0,6)})  last:${c.last_seen||"-"}`;
    el.appendChild(div);
  });
});

socket.on("processed", (d)=> {
  if(d.image) document.getElementById("proc_img").src = d.image;
  if(d.proc_ms!==undefined) document.getElementById("proc_info").innerText = "Proc: "+d.proc_ms+" ms";
});

socket.on("det_counts", (d)=>{
  let html = "";
  for(const k in d.counts){
    html += `<b>${k}</b>: ${d.counts[k]} &nbsp; `;
  }
  document.getElementById("proc_info").innerHTML = html;
});

// Fire alert handler: show banner, play sound, popup, red border
socket.on("fire_alert", (d)=>{
  try{
    const banner = document.getElementById("fire_banner");
    banner.style.display = "block";
    const alarm = document.getElementById("alarm_sound");
    // try to play alarm; may be blocked until user interacts with page
    alarm.play().catch(()=>{ /* ignore */ });
    // red border on processed image
    const pimg = document.getElementById("proc_img");
    pimg.style.border = "5px solid red";
    // show popup
    alert("🔥 FIRE DETECTED!\\nCamera: " + d.camera + "\\nTime: " + d.time);
    // auto-hide after 6 seconds
    setTimeout(()=>{
      banner.style.display = "none";
      pimg.style.border = "2px solid #222";
    }, 6000);
  }catch(e){ console.log("fire alert handler error", e); }
});


// simple fps graph updating from /api/stats poll
const fpsCtx = document.getElementById("fpsChart").getContext("2d");
const fpsChart = new Chart(fpsCtx, {
  type:'line',
  data:{labels:Array(60).fill(''), datasets:[{label:'FPS', data:Array(60).fill(0), borderColor:'rgba(80,200,120,1)', tension:0.3}]},
  options:{animation:false, scales:{x:{display:false}, y:{beginAtZero:true}}}
});

async function pollStats(){
  try{
    const r = await fetch("/api/stats");
    const j = await r.json();
    // take first client FPS to show
    const keys = Object.keys(j);
    let val = 0;
    if(keys.length) val = Math.round((j[keys[0]].fps||0)*10)/10;
    fpsChart.data.datasets[0].data.push(val);
    fpsChart.data.datasets[0].data.shift();
    fpsChart.update();
  }catch(e){}
}
setInterval(pollStats, 1000);

async function startCamera(){
  try{
    localStream = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
    document.getElementById("local_video").srcObject = localStream;
    startSending();
    const camname = document.getElementById("camera_name").value || ("cam_"+Math.random().toString(36).slice(2,8));
    socket.emit("register", {camera_name: camname});
  }catch(e){ alert("Camera access error: "+e); }
}

function startSending(){
  if(sending) return;
  sending = true;
  const video = document.getElementById("local_video");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  captureInterval = setInterval(()=>{
    if(video.readyState < 2) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video,0,0);
    // scale
    const scaled = document.createElement("canvas");
    scaled.width = 640;
    scaled.height = Math.round(640 * (canvas.height / canvas.width));
    scaled.getContext("2d").drawImage(canvas,0,0,scaled.width,scaled.height);
    const b64 = scaled.toDataURL("image/jpeg", 0.7);
    const camname = document.getElementById("camera_name").value || null;
    socket.emit("frame", {image: b64, camera_name: camname});
  }, 1000/fpsSend);
}

function stopSending(){
  sending = false;
  if(captureInterval){ clearInterval(captureInterval); captureInterval = null; }
  if(localStream){ localStream.getTracks().forEach(t=>t.stop()); localStream = null; }
}

document.getElementById("start").onclick = startCamera;
document.getElementById("stop").onclick = stopSending;

document.getElementById("screenshot").onclick = async ()=>{
  if(!localStream){ alert("Start camera first"); return;}
  const video = document.getElementById("local_video");
  const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video,0,0);
  const b64 = canvas.toDataURL("image/jpeg",0.9);
  socket.emit("screenshot", {image: b64});
  socket.once("screenshot_ack", (d)=>{ if(d.ok) alert("Saved: "+d.path); else alert("Screenshot failed: "+(d.msg||"")); });
};

document.getElementById("start_rec").onclick = ()=>{
  if(!localStream){ alert("Start camera"); return; }
  const video = document.getElementById("local_video");
  socket.emit("start_record", {width: video.videoWidth||640, height: video.videoHeight||480, fps: fpsSend});
  socket.once("record_started", (d)=>{ if(d.ok) alert("Recording started: "+d.path); else alert("Start failed: "+(d.msg||"")); });
};

document.getElementById("stop_rec").onclick = ()=>{
  socket.emit("stop_record", {});
  socket.once("record_stopped", (d)=>{ if(d.ok) alert("Saved: "+d.path); else alert("Stop failed: "+(d.msg||"")); });
};

document.getElementById("enroll").onclick = ()=>{
  const name = document.getElementById("enroll_name").value.trim();
  if(!name){ alert("Enter enroll name"); return; }
  const video = document.getElementById("local_video");
  if(!video || video.readyState<2){ alert("No frame"); return; }
  const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video,0,0);
  const b64 = canvas.toDataURL("image/jpeg",0.9);
  socket.emit("enroll_face", {name: name, image: b64});
  socket.once("enroll_ack", (d)=>{ alert(d.ok? "Enrolled":"Enroll failed: "+(d.msg||"")); });
};

</script>
</body>
</html>
"""

ADMIN_PAGE_HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Admin — Ultimate Multi-Cam</title></head>
<body style="background:#0b0b0b;color:#eee;font-family:Arial;padding:12px">
<h2>Admin Panel</h2>
<form id="auth">
  Password: <input id="pw" type="password">
  <button type="button" onclick="auth()">Login</button>
</form>
<div id="panel" style="display:none">
  <h3>Clients</h3>
  <pre id="clients"></pre>
  <h3>Timelines</h3>
  <pre id="timelines"></pre>
  <h3>Download</h3>
  <a href="/download_log">Download detection CSV</a>
  <h3>Recordings</h3>
  <div id="recs"></div>
</div>
<script>
let token = null;
function auth(){
  const pw = document.getElementById("pw").value;
  fetch("/api/clients?pw="+encodeURIComponent(pw)).then(r=>{
    if(r.status==200) return r.json();
    throw "auth failed";
  }).then(j=>{
    document.getElementById("panel").style.display="block";
    document.getElementById("clients").innerText = JSON.stringify(j, null, 2);
    token = pw;
    loadTimelines();
    loadRecs();
  }).catch(e=>alert("Auth failed"));
}
function loadTimelines(){
  fetch("/api/timelines").then(r=>r.json()).then(j=>{ document.getElementById("timelines").innerText = JSON.stringify(j,null,2); });
}
function loadRecs(){
  fetch("/api/clients?pw="+encodeURIComponent(token)).then(r=>r.json()).then(list=>{
    fetch("/").then(()=>{}); // noop
  });
  fetch("/").then(()=>{});
  fetch("/").then(()=>{});
}
</script>
</body>
</html>
"""

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("[INFO] Starting webrtc_full on port", port)
    socketio.run(app, host="0.0.0.0", port=port)
