from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from PIL import Image
import torch
import urllib.request
import os
import json
from typing import List

app = FastAPI(title="Simple AI Web App")

# Mount static folder (optional, for images or JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load pretrained lightweight model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(LABELS_URL, "imagenet_classes.txt")
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Server-side history file
HISTORY_FILE = "predictions.json"

def load_server_history() -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []

def save_server_history(items: List[dict]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(items, fh)
    except Exception as e:
        print("Failed to save history:", e)

# Route configurations for the app below


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the simple HTML interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Image Classifier</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root{
                --bg-1: #f7fbff;
                --bg-2: #eef6ff;
                --card: rgba(255,255,255,0.85);
                --accent: #6c8cff;
                --accent-600: #4f6fe6;
                --muted: #6b7280;
                --success: #dff3e1;
            }
            html,body{height:100%;}
            body {
                font-family: Inter, Arial, sans-serif;
                background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
                margin: 0;
                padding: 28px;
                color: #0f172a;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 1fr 320px;
                gap: 20px;
                align-items: start;
            }
            .card {
                background: var(--card);
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 6px 20px rgba(14, 30, 37, 0.06);
            }
            h1{margin:0 0 12px 0;font-size:20px}
            #drop-area{
                border: 2px dashed rgba(108,140,255,0.35);
                padding: 18px;
                border-radius:12px;
                text-align:center;
                transition:all .15s ease;
                min-height:260px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                gap:12px;
            }
            #drop-area.drag-over{box-shadow:0 8px 30px rgba(76,95,240,0.08); transform:translateY(-4px);}
            .muted{color:var(--muted)}
            #imagePreview img{max-width:100%; border-radius:10px; box-shadow:0 4px 18px rgba(16,24,40,0.06)}
            .controls{display:flex;gap:10px;justify-content:center;margin-top:8px}
            button{
                padding:10px 14px;border-radius:10px;border:none;cursor:pointer;font-weight:600;
                background:var(--accent);color:white;box-shadow:0 6px 12px rgba(79,111,230,0.12)
            }
            button.secondary{background:transparent;color:var(--accent-600);border:1px solid rgba(79,111,230,0.12)}
            button[disabled]{opacity:.5;cursor:not-allowed}
            .right-col{display:flex;flex-direction:column;gap:12px}
            .predicted-list{max-height:560px;overflow:auto;padding:6px;display:flex;flex-direction:column;gap:8px}
            .pred-item{display:flex;gap:8px;align-items:center;padding:8px;border-radius:10px;background:rgba(0,0,0,0.02)}
            .pred-item img{width:56px;height:56px;object-fit:cover;border-radius:8px}
            .pred-meta{flex:1}
            .result-area{margin-top:12px;padding:10px;border-radius:8px;background:var(--success);color:#064e2f}
            small.note{display:block;margin-top:6px;color:var(--muted)}
            @media(max-width:880px){.container{grid-template-columns:1fr;}.right-col{order:2}}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Image Classification</h1>
                <div id="drop-area" aria-label="Drop area">
                    <div class="muted">Drag & drop an image here or</div>
                    <div>
                        <input type="file" id="fileInput" accept="image/*" style="display:none;">
                        <button id="chooseBtn" class="secondary">Choose file</button>
                    </div>
                    <div id="imagePreview" aria-live="polite"></div>
                    <div class="controls">
                        <button id="predictButton" disabled>Predict</button>
                        <button id="clearButton" class="secondary">Clear</button>
                    </div>
                        <div style="display:flex;gap:10px;align-items:center;justify-content:center;margin-top:8px">
                            <div id="spinner" style="width:22px;height:22px;border-radius:50%;border:3px solid rgba(79,111,230,0.2);border-top-color:var(--accent-600);animation:spin 1s linear infinite;display:none"></div>
                            <div id="status" class="muted">No prediction yet.</div>
                        </div>
                        <div id="result" class="result-area" style="display:none;">
                            <strong id="predictionLabel"></strong>
                            <div><small id="predictionTime" class="muted"></small></div>
                            <ul id="topList" style="margin-top:8px"></ul>
                        </div>
                </div>
            </div>

            <div class="right-col">
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <strong>Previous Predictions</strong>
                        <button id="clearHistory" class="secondary">Clear all</button>
                    </div>
                    <small class="muted">Stored locally in your browser.</small>
                    <div id="predictedList" class="predicted-list"></div>
                </div>
                <div class="card muted">
                    <strong>How it works</strong>
                    <p class="muted">Images are sent to the server model for classification. The last predictions are kept locally for convenience.</p>
                </div>
            </div>
        </div>

        <script>
            // Elements
            const dropArea = document.getElementById('drop-area');
            const fileInput = document.getElementById('fileInput');
            const chooseBtn = document.getElementById('chooseBtn');
            const imagePreview = document.getElementById('imagePreview');
            const predictButton = document.getElementById('predictButton');
            const clearButton = document.getElementById('clearButton');
            const status = document.getElementById('status');
            const resultBox = document.getElementById('result');
            const predictionLabel = document.getElementById('predictionLabel');
            const predictionTime = document.getElementById('predictionTime');
            const predictedList = document.getElementById('predictedList');
            const clearHistory = document.getElementById('clearHistory');

            let currentDataUrl = null;

            // Utility: convert dataURL to Blob
            function dataURLtoBlob(dataurl) {
                const arr = dataurl.split(',');
                const mime = arr[0].match(/:(.*?);/)[1];
                const bstr = atob(arr[1]);
                let n = bstr.length;
                const u8arr = new Uint8Array(n);
                while(n--){ u8arr[n] = bstr.charCodeAt(n); }
                return new Blob([u8arr], {type:mime});
            }

            // Spinner helpers
            const spinner = document.getElementById('spinner');
            function showSpinner(){ spinner.style.display='block'; }
            function hideSpinner(){ spinner.style.display='none'; }

            // Rendering previous predictions
            function loadHistory(){
                try{
                    const raw = localStorage.getItem('predictions')||'[]';
                    return JSON.parse(raw);
                }catch(e){return[]}
            }

            function saveHistory(items){ localStorage.setItem('predictions', JSON.stringify(items)); }

            // Try to fetch server-side history; fallback to localStorage
            async function fetchServerHistory(){
                try{
                    const res = await fetch('/history');
                    if(!res.ok) throw new Error('no');
                    const data = await res.json();
                    if(data && Array.isArray(data.history)) return data.history;
                }catch(e){ return null; }
                return null;
            }

            async function renderServerHistory(){
                const items = await fetchServerHistory();
                if(items === null){ renderHistory(); return; }
                // render server items
                predictedList.innerHTML = '';
                if(!items.length){ predictedList.innerHTML = '<div class="muted">No saved predictions yet.</div>'; return }
                items.slice().reverse().forEach((it)=>{
                    const el = document.createElement('div'); el.className='pred-item';
                    const thumb = document.createElement('img'); thumb.src = it.dataUrl || '';
                    el.appendChild(thumb);
                    const meta = document.createElement('div'); meta.className='pred-meta';
                    const lbl = document.createElement('div'); lbl.textContent = it.label; meta.appendChild(lbl);
                    const when = document.createElement('small'); when.className='muted'; when.textContent = new Date(it.t).toLocaleString(); meta.appendChild(when);
                    el.appendChild(meta);
                    const actions = document.createElement('div');
                    const viewBtn = document.createElement('button'); viewBtn.textContent='Use'; viewBtn.style.padding='6px 8px'; viewBtn.onclick=()=>{useHistoryItem(it)};
                    const delBtn = document.createElement('button'); delBtn.textContent='Remove'; delBtn.className='secondary'; delBtn.style.padding='6px 8px'; delBtn.onclick=async()=>{ await fetch('/history', {method:'DELETE'}); renderServerHistory(); };
                    actions.appendChild(viewBtn); actions.appendChild(delBtn); el.appendChild(actions);
                    predictedList.appendChild(el);
                });
            }

            function renderHistory(){
                const items = loadHistory();
                predictedList.innerHTML = '';
                if(!items.length){ predictedList.innerHTML = '<div class="muted">No saved predictions yet.</div>'; return }
                items.slice().reverse().forEach((it)=>{
                    const el = document.createElement('div'); el.className='pred-item';
                    const thumb = document.createElement('img'); thumb.src = it.dataUrl || '';
                    el.appendChild(thumb);
                    const meta = document.createElement('div'); meta.className='pred-meta';
                    const lbl = document.createElement('div'); lbl.textContent = it.label; meta.appendChild(lbl);
                    const when = document.createElement('small'); when.className='muted'; when.textContent = new Date(it.t).toLocaleString(); meta.appendChild(when);
                    el.appendChild(meta);
                    const actions = document.createElement('div');
                    const viewBtn = document.createElement('button'); viewBtn.textContent='Use'; viewBtn.style.padding='6px 8px'; viewBtn.onclick=()=>{useHistoryItem(it)};
                    const delBtn = document.createElement('button'); delBtn.textContent='Remove'; delBtn.className='secondary'; delBtn.style.padding='6px 8px'; delBtn.onclick=()=>{removeHistoryItem(it.t)};
                    actions.appendChild(viewBtn); actions.appendChild(delBtn); el.appendChild(actions);
                    predictedList.appendChild(el);
                });
            }

            function removeHistoryItem(timestamp){
                let items = loadHistory(); items = items.filter(i=>i.t !== timestamp); saveHistory(items); renderHistory();
            }

            function useHistoryItem(item){
                // set preview from dataUrl and enable predict
                currentDataUrl = item.dataUrl;
                imagePreview.innerHTML = '<img src="'+item.dataUrl+'">';
                predictButton.disabled = false;
            }

            // Choose file button
            chooseBtn.addEventListener('click', ()=> fileInput.click());
            fileInput.addEventListener('change', (e)=>{
                handleFiles(e.target.files);
            });

            // Drag & drop handlers
            dropArea.addEventListener('dragover', (e)=>{ e.preventDefault(); dropArea.classList.add('drag-over'); });
            dropArea.addEventListener('dragleave', ()=> dropArea.classList.remove('drag-over'));
            dropArea.addEventListener('drop', (e)=>{
                e.preventDefault(); dropArea.classList.remove('drag-over');
                const files = e.dataTransfer.files; if(!files || !files.length) return;
                // create DataTransfer to set fileInput.files compatibly
                try{
                    const dt = new DataTransfer(); dt.items.add(files[0]); fileInput.files = dt.files;
                }catch(_){ /* fallback */ fileInput.files = files; }
                handleFiles(fileInput.files);
            });

            function handleFiles(files){
                const file = files[0];
                if(!file) return;
                const reader = new FileReader();
                reader.onload = function(ev){
                    currentDataUrl = ev.target.result;
                    imagePreview.innerHTML = '<img src="'+currentDataUrl+'">';
                    predictButton.disabled = false;
                    status.textContent = 'Ready to predict';
                    resultBox.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }

            // Predict function: uses either fileInput.files[0] if present, or currentDataUrl as blob
            async function predict(){
                predictButton.disabled = true;
                status.textContent = 'Predicting ...';
                try{
                    const form = new FormData();
                    if(fileInput.files && fileInput.files[0]){
                        form.append('file', fileInput.files[0]);
                    }else if(currentDataUrl){
                        const blob = dataURLtoBlob(currentDataUrl);
                        form.append('file', blob, 'image.png');
                    }else{
                        alert('No image selected'); predictButton.disabled=false; status.textContent='No image selected'; return;
                    }
                    const res = await fetch('/predict/', { method:'POST', body: form });
                    if(!res.ok) throw new Error('Server error');
                    const data = await res.json();
                    predictionLabel.textContent = data.predicted_class;
                    predictionTime.textContent = new Date().toLocaleString();
                    // list confidences
                    const topList = document.getElementById('topList'); topList.innerHTML = '';
                    if(data.predictions && Array.isArray(data.predictions)){
                        data.predictions.forEach(p=>{
                            const li = document.createElement('li');
                            li.textContent = `${p.label} — ${(p.score*100).toFixed(1)}%`;
                            topList.appendChild(li);
                        });
                    }
                    resultBox.style.display = 'block';
                    status.textContent = 'Done';

                    // Save to server-side history (fallback to localStorage)
                    const entry = { t: Date.now(), label: data.predicted_class, dataUrl: currentDataUrl };
                    try{
                        fetch('/history', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(entry)}).then(r=>{
                            if(!r.ok) throw new Error('server');
                            renderServerHistory();
                        }).catch(_=>{
                            const items = loadHistory(); items.push(entry); saveHistory(items); renderHistory();
                        });
                    }catch(_){ const items = loadHistory(); items.push(entry); saveHistory(items); renderHistory(); }
                }catch(err){
                    console.error(err); status.textContent = 'Prediction failed'; alert('Prediction failed: '+err.message);
                }finally{
                    predictButton.disabled = false;
                }
            }

            // Clear preview
            function clearPreview(){
                fileInput.value = '';
                currentDataUrl = null;
                imagePreview.innerHTML = '';
                predictButton.disabled = true;
                status.textContent = 'No prediction yet.';
                resultBox.style.display = 'none';
            }

            // Clear history
            function clearAllHistory(){ if(confirm('Clear all saved predictions?')){ localStorage.removeItem('predictions'); renderHistory(); }}

            // Wire up events
            predictButton.addEventListener('click', predict);
            clearButton.addEventListener('click', clearPreview);
            clearHistory.addEventListener('click', async ()=>{
                // try server-side clear first
                try{
                    const res = await fetch('/history', {method: 'DELETE'});
                    if(res.ok){ renderServerHistory(); return; }
                }catch(e){}
                // fallback
                clearAllHistory();
            });

            // Init
            renderHistory();
        </script>
    </body>
    </html>
    """


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return top-k predictions with confidence scores."""
    image = Image.open(file.file).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        topk = probs.topk(3)
        confidences = topk.values[0].cpu().tolist()
        indices = topk.indices[0].cpu().tolist()

    top_predictions = []
    for idx, score in zip(indices, confidences):
        top_predictions.append({
            "label": labels[idx],
            "score": float(score)
        })

    # Return both primary label and top-k list
    return JSONResponse({"predicted_class": top_predictions[0]["label"], "predictions": top_predictions})


@app.get("/history")
async def get_history():
    """Return server-side stored prediction history."""
    items = load_server_history()
    return JSONResponse({"history": items})


@app.post("/history")
async def post_history(request: Request):
    """Save a prediction entry to server-side history.

    Expects JSON: { label: str, dataUrl: str (optional), t: int }
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(payload, dict) or "label" not in payload:
        return JSONResponse({"error": "Missing label"}, status_code=400)

    items = load_server_history()
    items.append(payload)
    # keep only recent 200 entries to avoid unbounded growth
    if len(items) > 200:
        items = items[-200:]
    save_server_history(items)
    return JSONResponse({"ok": True})


@app.delete("/history")
async def delete_history():
    """Clear server-side history file."""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"ok": True})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Handle image upload and return prediction."""
    if file:
        # Process the uploaded file
        contents = await file.read()
        # Add your processing logic here
        return JSONResponse(content={"message": "File uploaded successfully!"})
    return JSONResponse(content={"error": "No file uploaded"}, status_code=400)
