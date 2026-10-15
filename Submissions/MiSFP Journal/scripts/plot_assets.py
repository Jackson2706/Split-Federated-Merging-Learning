#!/usr/bin/env python3
"""Generate deterministic explanatory figures and empty empirical templates."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, GRAY, RED = "#3569A8", "#D97732", "#3D8B66", "#697386", "#B64646"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

def multimodality():
    x = np.linspace(-4.5, 4.5, 600)
    normal = lambda x,m,s: np.exp(-.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))
    mix = .5*normal(x,-2,.5)+.5*normal(x,2,.5)
    single = normal(x,0,np.sqrt(4.25))
    fig, ax = plt.subplots(figsize=(3.45,2.15))
    ax.plot(x,mix,color=BLUE,lw=2,label="Two-component mixture")
    ax.plot(x,single,color=ORANGE,lw=2,ls="--",label="Moment-matched Gaussian")
    ax.fill_between(x,mix,alpha=.12,color=BLUE); ax.axvspan(-.6,.6,color=RED,alpha=.08)
    ax.text(0,.38,"low-density gap",ha="center",color=RED)
    ax.set(xlabel="$z_1$ (illustrative feature coordinate)",ylabel="Density",yticks=[])
    ax.legend(frameon=False,loc="upper right"); ax.set_title("Illustrative example")
    save(fig,"misfp_multimodality.pdf")

def box(ax,xy,w,h,text,color):
    p=FancyBboxPatch(xy,w,h,boxstyle="round,pad=.02",fc=color+"18",ec=color,lw=1.3)
    ax.add_patch(p); ax.text(xy[0]+w/2,xy[1]+h/2,text,ha="center",va="center"); return p
def arrow(ax,a,b,color=GRAY,ls="-",label=None,yoff=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,color=color,lw=1.3,ls=ls))
    if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+yoff,label,ha="center",color=color,fontsize=7)

def architecture():
    fig,ax=plt.subplots(figsize=(7.2,3.05)); ax.set(xlim=(0,12),ylim=(0,5)); ax.axis("off")
    box(ax,(.2,2.9),1.6,1,"Real $x,y$\nClient $u_i$",BLUE)
    box(ax,(2.4,2.9),1.7,1,"local class\nmixture",ORANGE)
    box(ax,(4.8,2.45),2.1,1.9,"Edge\nretain/compress\nsample $\tilde z$\ntrain $h_e$",GREEN)
    box(ax,(7.6,2.9),1.8,1,"output-space\nmixture",ORANGE)
    box(ax,(10.1,2.45),1.7,1.9,"Cloud\npool/sample\ntrain $g$",GREEN)
    arrow(ax,(1.8,3.4),(2.4,3.4),BLUE,label="features stay local",yoff=.25)
    arrow(ax,(4.1,3.4),(4.8,3.4),ORANGE,label="C--E packet",yoff=.25)
    arrow(ax,(6.9,3.4),(7.6,3.4),GREEN,ls="--",label="$h_e(\tilde z)$",yoff=.25)
    arrow(ax,(9.4,3.4),(10.1,3.4),ORANGE,label="E--C packet",yoff=.25)
    ax.text(5.85,1.7,"synthetic input; gradients local to edge",ha="center",color=GREEN)
    ax.text(10.95,1.7,"synthetic input; gradients local to cloud",ha="center",color=GREEN)
    arrow(ax,(1.0,2.6),(5.7,1.05),GRAY,ls="--",label="slow inherited same-tier model synchronization",yoff=-.2)
    ax.text(6,.25,"No per-example activation or gradient message crosses a split boundary",ha="center",weight="bold")
    save(fig,"misfp_architecture.pdf")

def merging():
    fig,ax=plt.subplots(figsize=(3.45,2.45)); ax.set(xlim=(0,10),ylim=(0,6)); ax.axis("off")
    pts=[(1,4.6,BLUE,"$c=1$"),(2,4.2,BLUE,"$c=1$"),(1.5,2.8,ORANGE,"$c=2$"),(3,4.8,BLUE,"$c=1$"),(3.2,2.4,ORANGE,"$c=2$")]
    for x,y,c,t in pts: ax.add_patch(Circle((x,y),.28,fc=c+"55",ec=c)); ax.text(x,y-.55,t,ha="center",fontsize=7)
    arrow(ax,(3.8,3.7),(5.2,3.7),GRAY,label="same-class\ngreedy merge",yoff=.55)
    for x,y,c,t in [(6,4.4,BLUE,"$c=1$"),(7.2,4.0,BLUE,"$c=1$"),(6.5,2.6,ORANGE,"$c=2$")]:
        ax.add_patch(Circle((x,y),.36,fc=c+"55",ec=c,lw=1.4)); ax.text(x,y-.65,t,ha="center",fontsize=7)
    arrow(ax,(7.8,3.7),(8.8,3.7),GRAY,label="budget",yoff=.35)
    ax.add_patch(Circle((9.3,4.2),.42,fc=BLUE+"55",ec=BLUE)); ax.add_patch(Circle((9.3,2.7),.36,fc=ORANGE+"55",ec=ORANGE))
    ax.text(5,5.7,"component IDs are local; class labels persist",ha="center",weight="bold")
    ax.text(5,.45,"Mass and diagonal moments are preserved; modes need not be",ha="center",color=RED)
    save(fig,"misfp_merging.pdf")

def pending(ax,title,xlabel,ylabel):
    ax.set_title(title,fontsize=9); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(.5,.5,"Results pending",transform=ax.transAxes,ha="center",va="center",color=GRAY,
            bbox=dict(boxstyle="round",fc="white",ec=GRAY,ls="--"))
    ax.grid(alpha=.2)
def templates():
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.9))
    pending(axs[0,0],"(a) Performance--communication","Cumulative communication (MiB)","Top-1 accuracy (%)")
    pending(axs[0,1],"(b) Convergence","Round or wall-clock time","Accuracy / macro-F1 (%)")
    pending(axs[1,0],"(c) Real vs. synthetic fidelity","Projection axis 1","Projection axis 2")
    pending(axs[1,1],"(d) Adaptive allocation","Components per class","Payload (MiB) / quality")
    fig.tight_layout(); save(fig,"empirical_templates.pdf")
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.9))
    pending(axs[0,0],"(a) Component cap","Maximum components/class","Accuracy / SWD")
    pending(axs[0,1],"(b) Minimum support","Minimum component support (samples)","Accuracy / mode coverage")
    pending(axs[1,0],"(c) Boundary Monte Carlo","Samples/component","SWD / runtime (s)")
    pending(axs[1,1],"(d) Encoder alignment","Extraction mode","Between-source dispersion")
    fig.tight_layout(); save(fig,"diagnostic_templates.pdf")

if __name__ == "__main__":
    multimodality(); architecture(); merging(); templates()
    (ROOT/"generated"/"asset_log.json").write_text(json.dumps({"explanatory":["misfp_multimodality.pdf","misfp_architecture.pdf","misfp_merging.pdf"],"placeholders":["empirical_templates.pdf","diagnostic_templates.pdf"]},indent=2)+"\n")
