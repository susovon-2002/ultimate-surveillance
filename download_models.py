# ultimate_timeline_wiki.py
# Clean, fixed, stable YOLO + WikiFusion + detection timeline

import os, time, threading, queue, requests, wikipedia, cv2, numpy as np
from collections import defaultdict, deque
from datetime import datetime
from flask import Flask, Response, jsonify, render_template_string

# ---------------- CONFIG ----------------
YOLO_MODEL = "yolov8n.pt"           # YOLOv8 nano (auto-download)
CAM_INDEX = 0                       # webcam index
PREVIEW_WIDTH = 480                # small preview for speed
CONF_THRESHOLD = 0.30
FRAME_SKIP = 2
STREAM_QUALITY = 75

# timeline settings
TIMELINE_LENGTH = 60               # samples in graph
TIMELINE_INTERVAL = 0.5            # polling speed (seconds)

# Queues
FRAME_Q = queue.Queue(maxsize=2)
OUT_Q = queue.Queue(maxsize=2)

# Caches and tracking
CACHE_TTL = 60*60*24
ASK_COOLDOWN = 2
INFO_CACHE = {}
LATEST_INFO = {}
LAST_ASK = defaultdict(lambda: 0)
DETECTION_COUNTER = defaultdict(int)
STATS = {"fps":0, "yolo_time":0}

# timeline array
timeline = deque([0]*TIMELINE_LENGTH, maxlen=TIMELINE_LENGTH)
timeline_lock = threading.Lock()

# Flask
app = Flask(__name__)


# ---------------- UTILS ----------------

def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, int(h * scale)))

def safe_wikipedia(q):
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(q, sentences=2)
    except:
        return None

def safe_wikidata(q):
    try:
        search = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": q,
                "language": "en",
                "format": "json"
            },
            timeout=5
        ).json()

        if not search["search"]:
            return None

        eid = search["search"][0]["id"]
        ent = requests.get(
            f"https://www.wikidata.org/wiki/Special:EntityData/{eid}.json",
            timeout=5
        ).json()["entities"][eid]

        desc = ent.get("descriptions", {}).get("en", {}).get("value", "")
        img = None
        if "P18" in ent.get("claims", {}):
            img_name = ent["claims"]["P18"][0]["mainsnak"]["datavalue"]["value"]
            img = "https://commons.wikimedia.org/wiki/Special:FilePath/" + img_name.replace(" ", "_")

        return {"label": q, "description": desc, "image": img}

    except:
        return None


def async_fetch_info(label):
    label = label.lower()
    now = time.time()

    # Cached → return
    if label in INFO_CACHE and now - INFO_CACHE[label]["ts"] < CACHE_TTL:
        LATEST_INFO[label] = INFO_CACHE[label]["data"]
        return

    # Cooldown
    if now - LAST_ASK[label] < ASK_COOLDOWN:
        return

    LAST_ASK[label] = now
    INFO_CACHE[label] = {"ts": now, "data": {"title": label, "summary": "Loading...", "description":"", "image": None}}
    LATEST_INFO[label] = INFO_CACHE[label]["data"]

    # Start background lookup
    def fetch():
        summary = safe_wikipedia(label)
        wd = safe_wikidata(label)
        data = {
            "title": label,
            "summary": summary or "",
            "description": wd["description"] if wd else "",
            "image": wd["image"] if (wd and wd.get("image")) else None
        }
        INFO_CACHE[label] = {"ts": time.time(), "data": data}
        LATEST_INFO[label] = data

    threading.Thread(target=fetch, daemon=True).start()


# ---------------- YOLO WRAPPER ----------------

class YOLOWrapper:
    def __init__(self, model_path):
        print("[INFO] Loading YOLO model...")

        from ultralytics import YOLO
        self.model = YOLO(model_path)

        # warm-up
        try:
            self.model.predict(np.zeros((320,320,3),dtype=np.uint8), imgsz=320, verbose=False)
        except:
            pass

        print("[INFO] YOLO loaded successfully.")

    def predict(self, frame):
        t0 = time.time()
        results = self.model.predict(frame, conf=CONF_THRESHOLD, imgsz=480, verbose=False)
        STATS["yolo_time"] = time.time() - t0
        return results


# ---------------- THREADS ----------------

def capture_thread():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    print("[INFO] Camera started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if FRAME_Q.full():
            try: FRAME_Q.get_nowait()
            except: pass

        FRAME_Q.put(frame)


def inference_thread(yolo):
    last_t = time.time()
    frame_id = 0

    while True:
        try:
            frame = FRAME_Q.get(timeout=1)
        except queue.Empty:
            continue

        frame_id += 1

        # skip frames for FPS boost
        if FRAME_SKIP > 1 and frame_id % FRAME_SKIP != 0:
            small = resize_to_width(frame, PREVIEW_WIDTH)
            push_out(small)
            continue

        results = yolo.predict(frame)

        ann = frame.copy()
        detected = 0
        labels = set()

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < CONF_THRESHOLD:
                    continue

                cls = int(box.cls)
                lbl = r.names[cls].lower()
                labels.add(lbl)
                DETECTION_COUNTER[lbl] += 1
                detected += 1

                # coords
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())

                # red box
                color = (0,0,255)
                cv2.rectangle(ann, (x1,y1), (x2,y2), color, 3)

                # red label bg
                txt = f"{lbl} {conf:.2f}"
                (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(ann, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
                cv2.putText(ann, txt, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)

        # timeline update
        with timeline_lock:
            timeline.append(detected)

        # wiki fetch (async)
        for lbl in labels:
            if lbl not in LATEST_INFO:
                async_fetch_info(lbl)

        # FPS
        now = time.time()
        STATS["fps"] = 1/(now-last_t+1e-9)
        last_t = now

        ann = resize_to_width(ann, PREVIEW_WIDTH)
        push_out(ann)


def push_out(frame):
    if OUT_Q.full():
        try: OUT_Q.get_nowait()
        except: pass
    try: OUT_Q.put_nowait(frame)
    except: pass


def mjpeg_stream():
    while True:
        try:
            frame = OUT_Q.get(timeout=1)
        except queue.Empty:
            continue

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() +
            b"\r\n"
        )


# ---------------- HTML ----------------

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>YOLO Timeline + WikiFusion</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{background:#0c0c0c;color:#eee;font-family:Arial;margin:0}
.wrap{display:flex;gap:10px;padding:10px}
.left{flex:1}
.right{width:360px;background:#121212;padding:10px;border-radius:8px}
img{width:100%;border-radius:10px}
.box{background:#1a1a1a;padding:10px;border-radius:6px;margin-bottom:10px}
.small{font-size:0.9em;color:#ccc}
</style>
</head>

<body>
<div class="wrap">
  <div class="left">
    <img src="/video" />
  </div>

  <div class="right">

    <div class="box">
      <strong>Timeline (objects per frame)</strong>
      <canvas id="timeline" height="160"></canvas>
    </div>

    <div class="box">
      <strong>Detected Objects</strong>
      <div id="detlist" class="small">Waiting...</div>
    </div>

    <div class="box">
      <strong>Object Info</strong>
      <div id="info" class="small">No info yet.</div>
    </div>

    <div class="box">
      <strong>Stats</strong>
      <div id="stats">...</div>
    </div>

  </div>
</div>

<script>
const LEN = {{length}};
let chart=null;

function initChart(){
  let ctx=document.getElementById("timeline").getContext("2d");
  chart=new Chart(ctx,{
    type:"line",
    data:{
      labels:Array(LEN).fill(""),
      datasets:[{
        label:"Objects",
        data:Array(LEN).fill(0),
        borderColor:"rgba(255,50,50,1)",
        backgroundColor:"rgba(255,50,50,0.2)",
        fill:true,
        tension:0.25
      }]
    },
    options:{
      animation:false,
      scales:{x:{display:false},y:{beginAtZero:true}}
    }
  });
}

async function updateTimeline(){
  let r=await fetch("/timeline");
  let arr=await r.json();
  chart.data.datasets[0].data=arr;
  chart.update();
}

async function updateInfo(){
  let r=await fetch("/info");
  let j=await r.json();
  let list=document.getElementById("detlist");
  let panel=document.getElementById("info");

  list.innerHTML="";
  let keys=Object.keys(j);

  if(keys.length===0){
    list.innerText="No detections.";
    panel.innerText="No info yet.";
    return;
  }

  keys.sort();
  for(let k of keys){
    let v=j[k];
    list.innerHTML+=`<div><b>${v.title}</b><br>${(v.summary||'').slice(0,150)}</div><br>`;
  }

  let last=keys[keys.length-1];
  let v=j[last];

  let html=`<b>${v.title}</b><p>${v.summary||''}</p><i>${v.description||''}</i>`;
  if(v.image){
    html+=`<div><img src="${v.image}" style="width:100%;border-radius:6px;margin-top:6px" /></div>`;
  }
  panel.innerHTML=html;
}

async function updateStats(){
  let r=await fetch("/stats");
  let s=await r.json();
  document.getElementById("stats").innerText=
    `FPS: ${s.fps.toFixed(1)}  |  YOLO: ${(s.yolo_time*1000).toFixed(1)} ms`;
}

initChart();
setInterval(updateTimeline, {{interval}});
setInterval(updateInfo, 1200);
setInterval(updateStats, 900);
</script>

</body>
</html>
"""

# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template_string(
        HTML, length=TIMELINE_LENGTH, interval=int(TIMELINE_INTERVAL*1000)
    )

@app.route("/video")
def video():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/timeline")
def timeline_route():
    with timeline_lock:
        return jsonify(list(timeline))

@app.route("/info")
def info():
    return jsonify(LATEST_INFO)

@app.route("/stats")
def stats():
    return jsonify(STATS)


# ---------------- MAIN ----------------

if __name__ == "__main__":

    # load YOLO safely
    yolo = YOLOWrapper(YOLO_MODEL)

    # start threads
    threading.Thread(target=capture_thread, daemon=True).start()
    threading.Thread(target=inference_thread, args=(yolo,), daemon=True).start()

    print("[INFO] Open: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)

