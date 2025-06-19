# app.py – “full” adaptive satellite/drone dehazing
import streamlit as st, cv2, numpy as np, io
from PIL import Image

st.set_page_config(page_title="Full Dehazing", layout="centered")
st.title("🛰️  Full Satellite / Drone Image Dehazing (Multi‑scale DCP)")

# ----------- adaptive presets -------------
PRESETS = {
    "thin":     dict(win=[15,35], omega=0.85, t0=0.06, pct=0.0008, r=30, eps=1e-3),
    "moderate": dict(win=[15,35,55], omega=0.9, t0=0.04, pct=0.0015, r=45, eps=1e-3),
    "thick":    dict(win=[35,55,75], omega=1.0, t0=0.02, pct=0.0025, r=60, eps=2e-3),
}

# ----------- core functions ---------------
def dark_channel(img, size):
    return cv2.erode(np.min(img,2), cv2.getStructuringElement(cv2.MORPH_RECT,(size,size)))

def multi_scale_dark(img, sizes):
    chans=[dark_channel(img,s) for s in sizes]
    return np.min(np.stack(chans,0),0)

def atm_light(img, dark, pct):
    h,w=dark.shape; n=max(int(h*w*pct),1)
    idx=np.unravel_index(np.argsort(dark.ravel())[-n:],(h,w))
    return np.mean(img[idx],0)

def transmission_est(img, A, omega, win):
    return 1-omega*multi_scale_dark(img/A, win)

def guided(I,p,r,eps):
    mI=cv2.boxFilter(I,-1,(r,r)); mp=cv2.boxFilter(p,-1,(r,r))
    cII=cv2.boxFilter(I*I,-1,(r,r)); cIp=cv2.boxFilter(I*p,-1,(r,r))
    varI=cII-mI*mI; cov=cIp-mI*mp
    a=cov/(varI+eps); b=mp-a*mI
    return cv2.boxFilter(a,-1,(r,r))*I+cv2.boxFilter(b,-1,(r,r))

def recover(img,A,t,t0): t=np.clip(t,t0,1)[:,:,None]; return np.clip((img-A)/t+A,0,1)

def post_contrast(rgb):
    # CLAHE + gamma 1.2
    lab=cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_RGB2LAB)
    L,a,b=cv2.split(lab)
    L=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(L)
    merged=cv2.merge((L,a,b))
    out=cv2.cvtColor(merged,cv2.COLOR_LAB2RGB)/255.0
    return np.power(out,1/1.2)

def full_dehaze(img_rgb, level="moderate"):
    p=PRESETS[level]
    def single_pass(im):
        I=im.astype(np.float32)/255.0
        dark=multi_scale_dark(I,p['win'])
        A=atm_light(I,dark,p['pct'])
        t0=transmission_est(I,A,p['omega'],p['win'])
        gray=cv2.cvtColor((I*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)/255.0
        t=guided(gray,t0,p['r'],p['eps'])
        J=recover(I,A,t,p['t0'])
        return (post_contrast(J)*255).astype(np.uint8), dark.mean()
    out,dc_avg=single_pass(img_rgb)
    # run a second pass if average dark‑channel still high
    if dc_avg>0.15:
        out,_=single_pass(out)
    return out

def resize(img,w=420):
    h,w0=img.shape[:2]; return cv2.resize(img,(w,int(h*w/w0)),cv2.INTER_AREA)

# ----------- Streamlit UI -----------------
level=st.selectbox("Haze density",["thin","moderate","thick"])
file =st.file_uploader("Upload hazy JPG/PNG",["jpg","jpeg","png"])

if file:
    bgr=cv2.imdecode(np.frombuffer(file.read(),np.uint8),cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("❌ Could not decode image.")
    else:
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        with st.spinner("Removing haze…"):
            out=full_dehaze(rgb,level)
        col1,col2=st.columns(2)
        col1.image(resize(rgb),caption="Original",use_container_width=True)
        col2.image(resize(out),caption="Dehazed",use_container_width=True)
        buf=io.BytesIO(); Image.fromarray(out).save(buf,format="JPEG")
        st.download_button("⬇️ Download",buf.getvalue(),"dehazed.jpg","image/jpeg")
else:
    st.info("Upload a hazy satellite / drone image to begin.")
