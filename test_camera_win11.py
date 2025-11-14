# ultimate_final_surveillance.py
"""
Ultimate Final Surveillance — Multi-Cam with:
- YOLOv8 detection (Ultralytics)
- Multi-camera (up to 4), robust Win11 capture (CAP_DSHOW)
- Red boxes, clickable labels via Detected list
- Scrolling timeline graph (objects/frame)
- WikiFusion (Wikipedia + Wikidata) info fetch (async)
- High-risk alerts (fire/knife/gun) -> beep + TTS + snapshot
- Face recognition (optional ONNX) and enrollment/attendance
- Screenshot button, CSV logs, threaded pipeline

Save as ultimate_final_surveillance.py and run.
"""

import os, time, threading, queue, requests, wikipedia, cv2, numpy as np, csv
from collections import deque, defaultdict
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request, send_file
import pyttsx3

# optional packages
try:
    import onnxruntime as ort
    ONNXRT = True
except Exception:
    ONNXRT = False

# Optional simpleaudio for beep; fallback silent
try:
    import simpleaudio as sa
    SIMPLEAUDIO = True
except Exception:
    SIMPLEAUDIO = False

# ---------------- CONFIG ----------------
CAM_SOURCES = [0]            # list: e.g. [0,1,"rtsp://..."], tested on Windows 11 with index 0
MAX_CAM = 4
CAM_SOURCES = CAM_SOURCES[:MAX_CAM]

YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")
PREVIEW_WIDTH = int(os.environ.get("PREVIEW_WIDTH", 480))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.30))
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 2))
STREAM_QUALITY = int(os.environ.get("STREAM_QUALITY", 75))
DEVICE = "cpu"   # ultralytics will default to CPU unless GPU configured

HIGH_RISK = {"fire", "knife", "gun", "weapon"}  # names lower-case, adjust per model labels

OUT_DIR = "ultimate_detections"
SNAP_DIR = os.path.join(OUT_DIR, "snapshots")
CLIP_DIR = os.path.join(OUT_DIR, "clips")
LOG_CSV = os.path.join(OUT_DIR, "detections_log.csv")
ATTEND_CSV = os.path.join(OUT_DIR, "attendance.csv")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(SNAP_DIR, exist_ok=True); os.makedirs(CLIP_DIR, exist_ok=True)

# face models (optional)
FACE_DETECTOR_ONNX = "models/ultraface_slim.onnx"
FACE_RECOG_ONNX = "models/mobilefacenet_arcface.onnx"

# timeline
TIMELINE_LENGTH = 60
TIMELINE_MS = 500  # timeline poll interval (ms)

# ---------------- GLOBALS ----------------
num_cams = len(CAM_SOURCES)
frame_queues = [queue.Queue(maxsize=2) for _ in range(num_cams)]
out_queues = [queue.Queue(maxsize=2) for _ in range(num_cams)]
latest_frames = {}   # cam_idx -> latest frame (RGB/BGR)
timeline_data = [deque([0]*TIMELINE_LENGTH, maxlen=TIMELINE_LENGTH) for _ in range(num_cams)]
timeline_locks = [threading.Lock() for _ in range(num_cams)]
latest_info = [{} for _ in range(num_cams)]
info_cache = [{} for _ in range(num_cams)]
last_ask = [defaultdict(lambda: 0.0) for _ in range(num_cams)]
detection_counters = [defaultdict(int) for _ in range(num_cams)]
stats_list = [{"fps":0.0,"yolo_time":0.0} for _ in range(num_cams)]

# face resources
face_session = None
face_rec_session = None
face_detector_available = False
face_rec_available = False
FACE_DB_DIR = "faces"
os.makedirs(FACE_DB_DIR, exist_ok=True)

# TTS engine
tts_engine = pyttsx3.init()
tts_lock = threading.Lock()

# ---------------- HELPERS ----------------
def nowts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_wikipedia(q):
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(q, sentences=2)
    except Exception:
        return None

def safe_wikidata(q):
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {"action":"wbsearchentities","search":q,"language":"en","format":"json"}
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        js = r.json()
        if "search" not in js or not js["search"]:
            return None
        eid = js["search"][0]["id"]
        ent = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{eid}.json", timeout=5).json()["entities"][eid]
        desc = ent.get("descriptions", {}).get("en", {}).get("value","")
        img = None
        if "P18" in ent.get("claims", {}):
            try:
                imgname = ent["claims"]["P18"][0]["mainsnak"]["datavalue"]["value"]
                img = "https://commons.wikimedia.org/wiki/Special:FilePath/" + imgname.replace(" ", "_")
            except Exception:
                img = None
        return {"label": q, "description": desc, "image": img}
    except Exception:
        return None

def append_log(row):
    header = ["timestamp","camera","class","title","summary","image"]
    exists = os.path.exists(LOG_CSV)
    try:
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        print("Log write err:", e)

def append_attendance(name):
    exists = os.path.exists(ATTEND_CSV)
    try:
        with open(ATTEND_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["timestamp","name"])
            w.writerow([datetime.now().isoformat(), name])
    except Exception as e:
        print("Attendance write err:", e)

# beep and TTS
def play_beep():
    if SIMPLEAUDIO:
        fs = 44100; f = 880; t = 0.25
        samples = (np.sin(2*np.pi*np.arange(int(fs*t))*f/fs)).astype(np.float32)
        audio = (samples * 32767).astype(np.int16)
        try:
            sa.play_buffer(audio, 1, 2, fs)
        except Exception:
            pass

def speak(text):
    try:
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    except Exception:
        pass

# save snapshot
def save_snapshot(cam_idx, img, prefix="alarm"):
    ts = nowts()
    fname = f"{prefix}_cam{cam_idx}_{ts}.jpg"
    path = os.path.join(SNAP_DIR, fname)
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print("Save snap err:", e)
    return path

# ---------- FACE ONNX helpers (optional) ----------
def load_face_models():
    global face_session, face_rec_session, face_detector_available, face_rec_available
    if not ONNXRT:
        return
    try:
        if os.path.exists(FACE_DETECTOR_ONNX):
            face_session = ort.InferenceSession(FACE_DETECTOR_ONNX, providers=["CPUExecutionProvider"])
            face_detector_available = True
            print("[INFO] Face detector loaded")
    except Exception as e:
        print("Face detector load fail:", e)
    try:
        if os.path.exists(FACE_RECOG_ONNX):
            face_rec_session = ort.InferenceSession(FACE_RECOG_ONNX, providers=["CPUExecutionProvider"])
            face_rec_available = True
            print("[INFO] Face recog loaded")
    except Exception as e:
        print("Face recog load fail:", e)

def get_face_embedding(crop):
    if not face_rec_available: return None
    try:
        inp = cv2.resize(crop, (112,112)).astype(np.float32)/255.0
        inp = np.transpose(inp, (2,0,1))[None, :].astype(np.float32)
        out = face_rec_session.run(None, {face_rec_session.get_inputs()[0].name: inp})
        emb = np.array(out[0][0])
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    except Exception:
        return None

def find_match(emb):
    best = -1; best_name = None
    for fn in os.listdir(FACE_DB_DIR):
        if not fn.endswith(".npy"): continue
        name = os.path.splitext(fn)[0]
        try:
            db = np.load(os.path.join(FACE_DB_DIR, fn))
            sim = float(np.dot(db, emb))
            if sim > best:
                best = sim; best_name = name
        except Exception:
            continue
    return best_name, best

def enroll_face_from_frame(name, frame):
    # center-crop heuristic
    h,w = frame.shape[:2]
    cx, cy = w//2, h//3
    size = min(w//3, h//3, 224)
    x1 = max(0, cx-size//2); y1 = max(0, cy-size//2)
    crop = frame[y1:y1+size, x1:x1+size]
    emb = get_face_embedding(crop)
    if emb is None:
        return False
    np.save(os.path.join(FACE_DB_DIR, f"{name}.npy"), emb)
    append_attendance(name)
    return True

# ---------- Async Wiki fetch ----------
def async_fetch_info(cam_idx, label):
    lk = label.lower().strip()
    now = time.time()
    cache = info_cache[cam_idx]
    last = last_ask[cam_idx]
    LATEST = latest_info[cam_idx]
    if lk in cache and now - cache[lk]["ts"] < 24*3600:
        LATEST[lk] = cache[lk]["data"]; return
    if now - last.get(lk,0) < 2.0:
        return
    last[lk] = now
    cache[lk] = {"ts": now, "data": {"title": label, "summary": "Loading...", "description":"", "image": None}}
    LATEST[lk][lk] = cache[lk]["data"]
    def bg():
        summary = safe_wikipedia(label)
        wd = safe_wikidata(label)
        data = {"title": label, "summary": summary or "", "description": (wd.get("description","") if wd else ""), "image": (wd.get("image") if wd else None)}
        cache[lk] = {"ts": time.time(), "data": data}
        LATEST[lk][lk] = data
        append_log([datetime.now().isoformat(), cam_idx, label, data.get("title",""), (data.get("summary") or "")[:200], data.get("image") or ""])
    threading.Thread(target=bg, daemon=True).start()

# ---------- Camera capture (robust for Win11) ----------
def capture_loop(cam_idx, src):
    print(f"[CAM {cam_idx}] Opening source: {src}")
    # proper backend for Windows: CAP_DSHOW
    cap = None
    try:
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
    except Exception as e:
        print(f"[CAM {cam_idx}] open error: {e}")
        try:
            cap = cv2.VideoCapture(src)
        except:
            cap = None

    retry = 0
    while (cap is None or not cap.isOpened()) and retry < 6:
        print(f"[CAM {cam_idx}] Retry open {retry+1}/6 ...")
        time.sleep(1)
        try:
            if isinstance(src, int):
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)
        except Exception as e:
            cap = None
        retry += 1

    if cap is None or not cap.isOpened():
        print(f"[CAM {cam_idx}] ERROR: cannot open source {src}")
        return

    print(f"[CAM {cam_idx}] Opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[CAM {cam_idx}] frame read fail -> reconnecting")
            try:
                cap.release()
            except: pass
            time.sleep(0.5)
            try:
                if isinstance(src, int):
                    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(src)
            except:
                time.sleep(0.5)
            continue

        # store latest frame and also push to queue for processing
        # resize to preview width (maintain aspect)
        h,w = frame.shape[:2]
        if w != PREVIEW_WIDTH:
            nh = int(h * (PREVIEW_WIDTH / float(w)))
            frame = cv2.resize(frame, (PREVIEW_WIDTH, nh))
        latest_frames[cam_idx] = frame.copy()
        try:
            if frame_queues[cam_idx].full(): _ = frame_queues[cam_idx].get_nowait()
            frame_queues[cam_idx].put_nowait(frame.copy())
        except Exception:
            pass
        time.sleep(0.003)

# ---------- YOLO inference per camera ----------
def inference_loop(cam_idx):
    # load model inside thread
    from ultralytics import YOLO
    try:
        model = YOLO(YOLO_MODEL)
        print(f"[CAM {cam_idx}] YOLO model loaded")
    except Exception as e:
        print(f"[CAM {cam_idx}] YOLO load failed: {e}")
        return

    frame_id = 0
    last_t = time.time()
    while True:
        try:
            frame = frame_queues[cam_idx].get(timeout=2.0)
        except queue.Empty:
            time.sleep(0.01); continue
        frame_id += 1
        if FRAME_SKIP > 1 and (frame_id % FRAME_SKIP) != 0:
            try:
                if out_queues[cam_idx].full(): _ = out_queues[cam_idx].get_nowait()
                out_queues[cam_idx].put_nowait(frame.copy())
            except Exception:
                pass
            continue

        t0 = time.time()
        try:
            results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=PREVIEW_WIDTH, device=DEVICE, verbose=False)
        except Exception as e:
            print(f"[CAM {cam_idx}] inference error: {e}")
            time.sleep(0.02); continue
        stats_list[cam_idx]["yolo_time"] = time.time() - t0

        ann = frame.copy()
        detected_count = 0
        detected_labels = set()

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None: continue
            for box in boxes:
                try:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD: continue
                    cls = int(box.cls[0])
                    name = r.names.get(cls, str(cls)).lower()
                    xy = box.xyxy[0].cpu().numpy().astype(int)
                except Exception:
                    try:
                        conf = float(box.conf)
                        cls = int(box.cls)
                        name = r.names.get(cls, str(cls)).lower()
                        xy = box.xyxy[0].numpy().astype(int)
                    except Exception:
                        continue

                x1,y1,x2,y2 = [int(v) for v in xy]
                detected_count += 1
                detected_labels.add(name)
                detection_counters[cam_idx][name] += 1

                # red box and label
                color = (0,0,255)
                cv2.rectangle(ann, (x1,y1), (x2,y2), color, 3)
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                by = max(0, y1 - th - 6)
                cv2.rectangle(ann, (x1, by), (x1+tw+6, y1), color, -1)
                cv2.putText(ann, label, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # face recognition quick heuristic if available and class is person
                if face_detector_available and face_rec_available and name in ("person","people"):
                    # crop top half of box
                    fh1 = y1
                    fh2 = y1 + max(20, (y2-y1)//2)
                    crop = ann[fh1:fh2, x1:x2]
                    if crop.size != 0:
                        emb = get_face_embedding(crop)
                        if emb is not None:
                            match_name, score = find_match(emb)
                            if match_name and score > 0.5:
                                txt2 = f"{match_name} {score:.2f}"
                                cv2.putText(ann, txt2, (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                                append_attendance(match_name)

                # high-risk action
                if name in HIGH_RISK:
                    snap = save_snapshot(cam_idx, ann, prefix="alarm")
                    append_log([datetime.now().isoformat(), cam_idx, name, "ALARM", "", snap])
                    threading.Thread(target=play_beep, daemon=True).start()
                    threading.Thread(target=speak, args=(f"Warning {name} detected on camera {cam_idx}",), daemon=True).start()

        # push timeline
        with timeline_locks[cam_idx]:
            timeline_data[cam_idx].append(detected_count)

        # async fetch info for found labels
        for lbl in detected_labels:
            async_fetch_info(cam_idx, lbl)

        # fps
        now = time.time()
        stats_list[cam_idx]["fps"] = 1.0 / (now - last_t + 1e-9)
        last_t = now

        # output annotated frame
        try:
            if out_queues[cam_idx].full(): _ = out_queues[cam_idx].get_nowait()
            out_queues[cam_idx].put_nowait(ann)
        except Exception:
            pass

# ---------- MJPEG generator ----------
def mjpeg_generator(cam_idx):
    while True:
        try:
            frame = out_queues[cam_idx].get(timeout=1.0)
        except queue.Empty:
            # fallback to latest_frames if available
            frame = latest_frames.get(cam_idx, None)
            if frame is None:
                time.sleep(0.01); continue
        ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
        if not ret:
            time.sleep(0.01); continue
        data = jpg.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

# ---------- Flask UI ----------
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ultimate Surveillance</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{background:#0b0b0b;color:#eee;font-family:Arial;margin:0;padding:10px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.cam{background:#111;padding:8px;border-radius:6px}
img{width:100%;border-radius:6px}
.rightcol{width:360px;position:fixed;right:10px;top:10px;background:#121212;padding:10px;border-radius:8px}
.btn{padding:8px;margin-top:6px;border-radius:6px;background:#222;color:#fff;border:0;cursor:pointer}
.small{font-size:0.9em;color:#bbb}
</style>
</head>
<body>
<h2>Ultimate Surveillance — Multi-Cam</h2>
<div style="display:flex;gap:10px;">
  <div style="flex:1">
    <div class="grid">
      {% for i in range(num_cams) %}
      <div class="cam">
        <h4>Camera {{i}}</h4>
        <img id="cam{{i}}" src="/stream/{{i}}">
        <div id="stat{{i}}" class="small">Cam {{i}} initializing...</div>
        <button class="btn" onclick="snapshot({{i}})">Screenshot</button>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="rightcol">
    <div><strong>Timeline (objects/frame)</strong><canvas id="timeline" height="160"></canvas></div>
    <hr/>
    <div><strong>Detected (latest)</strong><div id="detlist" class="small">No detections yet.</div></div>
    <hr/>
    <div><strong>Info</strong><div id="infopanel" class="small">No info yet.</div></div>
    <hr/>
    <div><strong>Controls</strong>
      <div><button class="btn" onclick="downloadLog()">Download Log CSV</button></div>
      <div style="margin-top:8px"><input id="enrollName" placeholder="Name to enroll" style="width:100%;padding:6px"/><button class="btn" onclick="enrollFace()">Enroll Face</button></div>
    </div>
  </div>
</div>

<script>
const NUM = {{num_cams}};
const LEN = {{len}};
let chart=null;

function initChart(){
  const ctx = document.getElementById('timeline').getContext('2d');
  chart = new Chart(ctx, {type:'line', data:{labels:Array(LEN).fill(''), datasets:[{label:'Objects/frame', data:Array(LEN).fill(0), borderColor:'rgba(255,50,50,1)', backgroundColor:'rgba(255,50,50,0.12)', fill:true}]}, options:{animation:false,scales:{x:{display:false},y:{beginAtZero:true}}}});
}

async function updateTimeline(){
  try{
    const r = await fetch('/timeline_all');
    const arr = await r.json();
    chart.data.datasets[0].data = arr;
    chart.update();
  } catch(e){ console.log(e); }
}

async function updateDetections(){
  try{
    const r = await fetch('/info_all');
    const j = await r.json();
    const list = document.getElementById('detlist'); list.innerHTML='';
    let keys = [];
    for(let c=0;c<NUM;c++){
      let cam = j[c]||{};
      for(let k in cam){ keys.push(k); }
    }
    keys = [...new Set(keys)];
    if(keys.length===0){ list.innerText = 'No detections yet.'; document.getElementById('infopanel').innerText='No info yet.'; return; }
    for(const k of keys){
      const div = document.createElement('div'); div.className='small';
      div.innerHTML = `<b>${k}</b> <button onclick="showInfo('${k}')" style="margin-left:8px">Details</button>`;
      list.appendChild(div);
    }
  } catch(e){ console.log(e); }
}

async function showInfo(lbl){
  try{
    const r = await fetch('/get_info_label?label='+encodeURIComponent(lbl));
    const j = await r.json();
    let html = `<b>${j.title}</b><p>${j.summary||''}</p><i>${j.description||''}</i>`;
    if(j.image) html += `<div><img src="${j.image}" style="width:100%;margin-top:6px;border-radius:6px"/></div>`;
    document.getElementById('infopanel').innerHTML = html;
  } catch(e){ console.log(e); }
}

async function updateStats(){
  try{
    const r = await fetch('/stats_all');
    const s = await r.json();
    for(let i=0;i<NUM;i++){
      const el = document.getElementById('stat'+i);
      const st = s[i]||{fps:0,yolo_time:0};
      el.innerText = `Cam ${i} FPS:${(st.fps||0).toFixed(1)}  Y_ms:${Math.round((st.yolo_time||0)*1000)}`;
    }
  } catch(e){}
}

function snapshot(cam){
  fetch('/screenshot?cam='+cam).then(r=>r.blob()).then(b=>{
    const url = URL.createObjectURL(b);
    const a = document.createElement('a'); a.href=url; a.download='screenshot_cam'+cam+'.jpg'; a.click();
  }).catch(e=>alert('Screenshot failed'));
}

function downloadLog(){ window.location='/download_log'; }

async function enrollFace(){
  const name = document.getElementById('enrollName').value.trim();
  if(!name){ alert('Enter a name'); return; }
  const r = await fetch('/enroll', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:name, cam:0})});
  const j = await r.json();
  alert(j.msg);
}

initChart();
setInterval(updateTimeline, {{interval}});
setInterval(updateDetections, 1200);
setInterval(updateStats, 900);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, num_cams=num_cams, len=TIMELINE_LENGTH, interval=TIMELINE_MS)

@app.route("/stream/<int:cid>")
def stream(cid):
    if 0 <= cid < num_cams:
        return Response(mjpeg_generator(cid), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Invalid camera", 404

@app.route("/timeline_all")
def timeline_all():
    # average across cams
    tot = [0]*TIMELINE_LENGTH
    for i in range(num_cams):
        with timeline_locks[i]:
            arr = list(timeline_data[i])
        for j,v in enumerate(arr):
            tot[j] += v
    avg = [v/max(1,num_cams) for v in tot]
    return jsonify(avg)

@app.route("/info_all")
def info_all():
    return jsonify(latest_info)

@app.route("/get_info_label")
def get_info_label():
    label = request.args.get("label","").strip().lower()
    for cam in latest_info:
        if label in cam:
            return jsonify(cam[label])
    return jsonify({"title":label,"summary":"","description":"","image":None})

@app.route("/stats_all")
def stats_all():
    return jsonify(stats_list)

@app.route("/screenshot")
def screenshot():
    cam = int(request.args.get("cam",0))
    if cam < 0 or cam >= num_cams: return "Invalid cam", 400
    frame = latest_frames.get(cam, None)
    if frame is None:
        return "No frame", 404
    path = save_snapshot(cam, frame, prefix="screenshot")
    return send_file(path, mimetype="image/jpeg", as_attachment=True, download_name=os.path.basename(path))

@app.route("/download_log")
def download_log():
    if os.path.exists(LOG_CSV):
        return send_file(LOG_CSV, as_attachment=True)
    return "No log", 404

@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.get_json() or {}
    name = data.get("name","").strip()
    cam = int(data.get("cam", 0))
    if not name:
        return jsonify({"ok":False, "msg":"Name required"})
    frame = latest_frames.get(cam, None)
    if frame is None:
        return jsonify({"ok":False, "msg":"No frame available"})
    ok = enroll_face_from_frame(name, frame)
    if ok:
        return jsonify({"ok":True, "msg":"Enrolled and attendance recorded"})
    else:
        return jsonify({"ok":False, "msg":"Enrollment failed (face models missing or error)."})

# ---------- start threads ----------
def start_threads():
    # capture threads
    for i,src in enumerate(CAM_SOURCES):
        tcap = threading.Thread(target=capture_loop, args=(i,src), daemon=True); tcap.start()
    # inference threads
    for i in range(num_cams):
        tinf = threading.Thread(target=inference_loop, args=(i,), daemon=True); tinf.start()

# ---------- main ----------
if __name__ == "__main__":
    print("[INFO] Ultimate Surveillance starting")
    if ONNXRT:
        load_face_models()
    else:
        print("[INFO] onnxruntime not installed — face features disabled")

    # start all threads
    start_threads()

    print("[INFO] Open http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
