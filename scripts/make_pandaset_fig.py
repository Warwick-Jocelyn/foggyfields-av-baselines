"""Clean conference-style PandaSet comparison: common-width vertical stack with label bars.
Handles the fact that EmerNeRF's panorama layout differs from nerfstudio's."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PO=Path("/networkhome/WMGDS/wang3_y/ETH/Code/outputs")
OUT=Path("/networkhome/WMGDS/wang3_y/ETH/foggyfields-av-baselines/assets"); OUT.mkdir(exist_ok=True)
TMP=Path("/networkhome/WMGDS/wang3_y/downloads/_fig_tmp"); TMP.mkdir(exist_ok=True)
W=1500; PAD=10; LABEL_H=34; TITLE_H=46
NAVY=(20,36,59); RED=(150,30,30); WHITE=(255,255,255)
def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            try: return ImageFont.truetype(p,sz)
            except Exception: pass
    return ImageFont.load_default()
F_T=font(26); F_L=font(20)

def frame(video,key):
    out=TMP/f"pf_{key}.jpg"
    if out.exists(): return Image.open(out).convert("RGB")
    if not Path(video).exists(): return None
    png=TMP/f"pf_{key}.png"
    for a in (["-ss","1","-i",str(video),"-frames:v","1"],["-i",str(video),"-frames:v","1"]):
        try: subprocess.run(["ffmpeg","-y",*a,"-q:v","2",str(png)],check=True,capture_output=True,timeout=60)
        except Exception: continue
        if png.exists(): break
    if not png.exists(): return None
    im=Image.open(png).convert("RGB"); im.save(out,"JPEG",quality=90); png.unlink(missing_ok=True); return im

CLIPS={
 "011_day":("PandaSet clip 011 — daytime (fog-free reference)",NAVY,[
   ("Ground Truth",NAVY,PO/"paper_011_split05_paperfaithful/videos/val/panorama_gt_rgb.mp4","gt11"),
   ("SplatAD   ·   PSNR 27.52",(46,92,181),PO/"paper_011_split05_paperfaithful/videos/val/panorama_rgb.mp4","s11"),
   ("NeuRAD   ·   PSNR 26.46",(46,92,181),PO/"neurad_paper_011_split05/videos/val/panorama_rgb.mp4","n11"),
   ("EmerNeRF   ·   PSNR 27.55  (own camera layout)",(27,122,61),PO/"emernerf_011_split05/foggyfields_emernerf/scene_011_6cam_dyn_flow_rgb/test_videos/30000_rgbs.mp4","e11")]),
 "078_night":("PandaSet clip 078 — NIGHTTIME (fog-free reference)",RED,[
   ("Ground Truth",NAVY,PO/"paper_078_split05_paperfaithful/videos/val/panorama_gt_rgb.mp4","gt78"),
   ("SplatAD   ·   PSNR 31.65",(46,92,181),PO/"paper_078_split05_paperfaithful/videos/val/panorama_rgb.mp4","s78"),
   ("NeuRAD   ·   PSNR 30.04",(46,92,181),PO/"neurad_paper_078_split05/videos/val/panorama_rgb.mp4","n78"),
   ("EmerNeRF   ·   PSNR 31.26  (own camera layout)",(27,122,61),PO/"emernerf_078_split05/foggyfields_emernerf/scene_078_6cam_dyn_flow_rgb/test_videos/30000_rgbs.mp4","e78")]),
}
for key,(title,tcol,rows) in CLIPS.items():
    imgs=[]
    for label,col,vid,k in rows:
        im=frame(vid,k)
        if im is None: continue
        h=int(im.height*W/im.width); im=im.resize((W,h))
        imgs.append((label,col,im))
    total_h=TITLE_H+sum(LABEL_H+im.height+PAD for _,_,im in imgs)+PAD
    canvas=Image.new("RGB",(W+2*PAD,total_h),WHITE); d=ImageDraw.Draw(canvas)
    d.text((PAD,10),title,font=F_T,fill=tcol)
    y=TITLE_H
    for label,col,im in imgs:
        d.rectangle([PAD,y,PAD+W,y+LABEL_H-4],fill=col)
        d.text((PAD+10,y+5),label,font=F_L,fill=WHITE)
        y+=LABEL_H
        canvas.paste(im,(PAD,y)); y+=im.height+PAD
    fp=OUT/f"pandaset_{key}.png"; canvas.save(fp,"PNG"); print("->",fp,f"{fp.stat().st_size/1e6:.2f} MB  {canvas.size}")
print("done")
