"""
CBG Plant Layout Planner · v10.0 (CAD Import + Process-Flow Optimizer)
────────────────────────────────────────────────────────────────────────
Upload a biogas/CBG plant DXF → site boundary + already-built equipment
(digesters, gas holder, etc.) auto-load and auto-classify → configure the
process equipment you still need to place, the process-flow connections
between stages, and hard safety clearances → click Generate Layouts →
CP-SAT solves a facility-layout problem (not bin-packing): it first
maximizes how much equipment fits, then produces several genuinely
different named layouts (flow-optimized, compact, balanced-pipe-run),
each guaranteed to respect every hard safety clearance.

NOTE: default equipment footprints, flow weights, and safety clearance
distances are engineering-judgment starting points, not verified
regulatory values. Review them against the codes that apply in your
jurisdiction (e.g. OISD/PESO in India, NFPA 30/58 elsewhere) before using
any generated layout for actual construction.

Run:   streamlit run app.py
Deps:  pip install streamlit ezdxf shapely ortools Pillow numpy pandas
"""

import json
import copy
import uuid

import numpy as np
import streamlit as st
import pandas as pd
from shapely.geometry import Polygon

from engine import rasterize, generate_layouts, BuildingSpec, FlowEdge, SafetyRule, Placement, LayoutResult
from dxf_import import import_dxf, DxfImportError, EquipmentZone
import pdf_import
import biogas_config as bc

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBG Plant Layout Planner · CAD + Flow Optimizer",
    page_icon="⛽",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CANVAS COMPONENT  (inline HTML / CSS / JS)
# ─────────────────────────────────────────────────────────────
# The SVG canvas renders boundary, obstacles, and buildings.
# All coordinates are in metres. It syncs state back to Python
# via Streamlit's component API on every user interaction.

_CANVAS_HTML = """
  <div id="toolbar">
    <div class="group">
      <label>Add shape:</label>
      <button class="shape-btn" draggable="true" data-shape="rectangle"><span class="shape-icon">&#9646;</span>Rect</button>
      <button class="shape-btn" draggable="true" data-shape="circle"><span class="shape-icon">&#9679;</span>Circle</button>
      <button class="shape-btn" draggable="true" data-shape="triangle"><span class="shape-icon">&#9650;</span>Tri</button>
      <button class="shape-btn" draggable="true" data-shape="l_shape"><span class="shape-icon">&#8990;</span>L</button>
    </div>
    <div class="group" style="margin-left:10px;padding-left:10px;border-left:1px solid #D4DFEA;">
      <label><input type="checkbox" id="snapGrid"> Grid Snap</label>
      <button class="shape-btn" id="clearBtn" style="color:#C0392B;">Clear All</button>
    </div>
    <div class="group" style="margin-left:auto;">
      <label>Boundary:</label>
      <select id="boundarySelect">
        <option value="custom">— from CAD —</option>
        <option value="rectangle">Rectangle</option>
        <option value="l_shape">L-Shape</option>
        <option value="pentagon">Pentagon</option>
        <option value="hexagon">Hexagon</option>
        <option value="trapezoid">Trapezoid</option>
      </select>
      <label>Sides:</label>
      <input type="number" id="boundarySides" min="3" max="20" step="1" style="width:48px" />
    </div>
  </div>
  <div class="hint">
    💡 <b>Pan:</b> drag background &nbsp;·&nbsp; <b>Zoom:</b> scroll &nbsp;·&nbsp;
    <b>Move:</b> drag building &nbsp;·&nbsp; <b>Resize:</b> corner handles &nbsp;·&nbsp;
    <b>Rotate:</b> blue circle handle
  </div>
  <div id="main">
    <div id="canvasWrap">
      <svg id="canvas" viewBox="0 0 200 130" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
    <div id="side">
      <h4>Site stats</h4>
      <div class="stats-row" id="statsRow"></div>
      <h4>Buildings</h4>
      <div id="bldList"></div>
    </div>
  </div>
"""

_CANVAS_CSS = """
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
  body { padding: 10px; background: #F7F9FC; color: #1F2D3D; }

  #toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 8px; }
  .group { display: flex; align-items: center; gap: 6px; }
  .group label { font-size: 12px; font-weight: 600; color: #4A5A6A; cursor: pointer; }

  .shape-btn {
    display: flex; align-items: center; gap: 4px;
    border: 1px solid #C7D4E2; background: #fff; border-radius: 6px;
    padding: 6px 10px; font-size: 12.5px; cursor: pointer; user-select: none;
    transition: box-shadow .15s, transform .15s;
  }
  .shape-btn:hover { box-shadow: 0 2px 6px rgba(0,0,0,.12); transform: translateY(-1px); }
  .shape-btn:active { transform: translateY(0); }
  .shape-icon { font-size: 13px; }

  select, input[type="text"], input[type="number"] {
    border: 1px solid #C7D4E2; border-radius: 6px; padding: 5px 8px; font-size: 12.5px;
  }

  .hint { font-size: 12px; color: #607182; margin-bottom: 10px; background: #EBF2FA;
          padding: 6px 10px; border-radius: 6px; display: inline-block; }

  #main { display: flex; gap: 14px; align-items: flex-start; height: 680px; }
  #canvasWrap {
    flex: 1 1 auto; height: 100%; border: 1px solid #C7D4E2; border-radius: 10px;
    background: #fff; min-width: 0; position: relative; overflow: hidden;
    box-shadow: inset 0 2px 8px rgba(0,0,0,.04);
  }
  svg#canvas { width: 100%; height: 100%; display: block; touch-action: none; background: #FAFBFC; }

  #side {
    flex: 0 0 300px; max-width: 300px; height: 100%;
    border: 1px solid #D4DFEA; border-radius: 10px; background: #fff; padding: 12px;
    overflow-y: auto; display: flex; flex-direction: column;
  }
  #side h4 { margin: 0 0 8px; font-size: 12px; color: #4A5A6A; text-transform: uppercase; letter-spacing: .5px; }

  .stats-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .stat { background: #F0F4FA; border-left: 3px solid #4A90D9; border-radius: 6px;
          padding: 6px 8px; font-size: 11px; flex: 1 1 75px; }
  .stat b { display: block; font-size: 14px; color: #1F2D3D; }

  #bldList { flex: 1 1 auto; overflow-y: auto; }
  .bld-row { border: 1px solid #E3EAF2; border-radius: 8px; padding: 8px; margin-bottom: 8px; background: #FFF; }
  .bld-row.selected { border-color: #4A90D9; background: #F3F8FE; box-shadow: 0 2px 6px rgba(74,144,217,.15); }
  .bld-row .top { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }
  .color-picker { width: 24px; height: 26px; padding: 0; border: 1px solid #C7D4E2; border-radius: 4px;
                  cursor: pointer; flex: none; background: none; }
  .bld-row input[type="text"] { flex: 1 1 auto; min-width: 0; font-size: 12.5px; padding: 3px 6px; }
  .icon-btn { border: none; background: #F0F4FA; color: #4A5A6A; border-radius: 4px;
              width: 26px; height: 26px; cursor: pointer; font-size: 13px; flex: none;
              display: flex; align-items: center; justify-content: center; }
  .icon-btn:hover { background: #E3EAF2; }
  .icon-btn.del { background: #FCEAEA; color: #C0392B; }
  .icon-btn.del:hover { background: #FAD4D4; }
  .bld-row .meta { font-size: 11px; color: #8094A8; margin-bottom: 4px; padding-left: 2px; }
  .dims-row { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #4A5A6A; margin-top: 4px; }
  .dims-row input[type="number"] { width: 48px; padding: 3px 4px; font-size: 11.5px;
                                    border: 1px solid #C7D4E2; border-radius: 4px; text-align: center; }
  .dims-row input[type="number"]:focus { outline: none; border-color: #4A90D9; }
  .dims-row .unit { color: #8094A8; margin-right: 4px; font-size: 10px; }
  .empty-list { font-size: 13px; color: #95A5B5; text-align: center; padding: 30px 4px; }
  .warn-badge { color: #E74C3C; font-weight: 600; }
  .obstacle-badge { background: #FFF3E0; color: #E65100; border-left: 3px solid #E65100;
                    border-radius: 5px; font-size: 11px; padding: 4px 8px; margin-bottom: 10px; }
"""

_CANVAS_JS = """
export default function(component) {
const { data, parentElement, setStateValue } = component;

const svg = parentElement.querySelector("#canvas");
if (!svg) return;
svg.__setStateValue = setStateValue;
if (svg.dataset.initialized === "true") return;
svg.dataset.initialized = "true";

// ── Constants ────────────────────────────────────────────────
const WORLD_W = 200, WORLD_H = 130;
const MIN_SIZE = 0.5;
const MAX_AREA = 2000000;
let SNAP = 1;

let viewX = 0, viewY = 0, viewW = WORLD_W, viewH = WORLD_H;
let STATE = { boundary: null, obstacles: [], buildings: [] };
let nextId = 1, selected = null, drag = null;

const bSel   = parentElement.querySelector("#boundarySelect");
const bSides = parentElement.querySelector("#boundarySides");
const bldListEl  = parentElement.querySelector("#bldList");
const statsRowEl = parentElement.querySelector("#statsRow");

// ── Helpers ──────────────────────────────────────────────────
function snap(v) { return Math.round(v / SNAP) * SNAP; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function updateViewBox() {
  svg.setAttribute("viewBox", `${viewX} ${viewY} ${viewW} ${viewH}`);
}

function fitViewToState() {
  if (!STATE.boundary || !STATE.boundary.points || STATE.boundary.points.length < 3) return;
  const bb = getBBox(STATE.boundary.points);
  const siteW = bb.maxX - bb.minX, siteH = bb.maxY - bb.minY;
  if (siteW < 0.01 || siteH < 0.01) return;
  const pad = Math.max(siteW, siteH) * 0.08;
  const fitW = siteW + 2 * pad, fitH = siteH + 2 * pad;
  const aspect = svg.clientWidth > 0 ? svg.clientWidth / svg.clientHeight : WORLD_W / WORLD_H;
  if (fitW / fitH > aspect) {
    viewW = fitW; viewH = fitW / aspect;
  } else {
    viewH = fitH; viewW = fitH * aspect;
  }
  viewX = bb.minX - pad - (viewW - fitW) / 2;
  viewY = bb.minY - pad - (viewH - fitH) / 2;
  updateViewBox();
}

function screenToWorld(clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX; pt.y = clientY;
  const ctm = svg.getScreenCTM();
  if (!ctm) return [clientX, clientY];
  const p = pt.matrixTransform(ctm.inverse());
  return [p.x, p.y];
}

function getBBox(pts) {
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for (const [x,y] of pts) {
    if(x<minX)minX=x; if(x>maxX)maxX=x;
    if(y<minY)minY=y; if(y>maxY)maxY=y;
  }
  return {minX,minY,maxX,maxY};
}

function rotatePt([px,py], cx, cy, deg) {
  if (!deg) return [px,py];
  const rad = deg*Math.PI/180, cos=Math.cos(rad), sin=Math.sin(rad);
  const dx=px-cx, dy=py-cy;
  return [cx+dx*cos-dy*sin, cy+dx*sin+dy*cos];
}

function shoelaceArea(pts) {
  let s=0;
  for(let i=0;i<pts.length;i++){
    const[x1,y1]=pts[i],[x2,y2]=pts[(i+1)%pts.length];
    s+=x1*y2-x2*y1;
  }
  return Math.abs(s)/2;
}

function pointInPolygon(px,py,pts){
  let inside=false;
  for(let i=0,j=pts.length-1;i<pts.length;j=i++){
    const[xi,yi]=pts[i],[xj,yj]=pts[j];
    if((yi>py)!==(yj>py)&&px<((xj-xi)*(py-yi))/(yj-yi)+xi)inside=!inside;
  }
  return inside;
}

function segmentsIntersect(p1,p2,p3,p4){
  const ccw=(a,b,c)=>(c[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(c[0]-a[0]);
  return ccw(p1,p3,p4)!==ccw(p2,p3,p4)&&ccw(p1,p2,p3)!==ccw(p1,p2,p4);
}

function polysOverlap(ptsA,ptsB){
  const bbA=getBBox(ptsA),bbB=getBBox(ptsB);
  if(bbA.maxX<bbB.minX||bbA.minX>bbB.maxX||bbA.maxY<bbB.minY||bbA.minY>bbB.maxY)return false;
  for(const p of ptsA)if(pointInPolygon(p[0],p[1],ptsB))return true;
  for(const p of ptsB)if(pointInPolygon(p[0],p[1],ptsA))return true;
  for(let i=0;i<ptsA.length;i++)for(let j=0;j<ptsB.length;j++)
    if(segmentsIntersect(ptsA[i],ptsA[(i+1)%ptsA.length],ptsB[j],ptsB[(j+1)%ptsB.length]))return true;
  return false;
}

// ── Boundary presets ─────────────────────────────────────────
const BOUNDARY_DEFAULT_CX=90, BOUNDARY_DEFAULT_CY=65, BOUNDARY_DEFAULT_R=55;
function genNGon(n,cx,cy,r){
  const pts=[];
  for(let i=0;i<n;i++){
    const a=-Math.PI/2+(2*Math.PI*i)/n;
    pts.push([snap(cx+r*Math.cos(a)),snap(cy+r*Math.sin(a))]);
  }
  return pts;
}
const BOUNDARY_PRESETS={
  rectangle:[[15,15],[165,15],[165,110],[15,110]],
  l_shape:  [[15,15],[165,15],[165,62],[90,62],[90,110],[15,110]],
  pentagon: genNGon(5,BOUNDARY_DEFAULT_CX,BOUNDARY_DEFAULT_CY,BOUNDARY_DEFAULT_R),
  hexagon:  genNGon(6,BOUNDARY_DEFAULT_CX,BOUNDARY_DEFAULT_CY,BOUNDARY_DEFAULT_R),
  trapezoid:[[51,15],[149,15],[165,110],[35,110]],
};

// ── Building shapes ──────────────────────────────────────────
const BUILDING_SHAPES={
  rectangle:{norm:[[0,0],[1,0],[1,1],[0,1]],            defW:24,defH:16,label:"Rectangle"},
  triangle: {norm:[[.5,0],[1,1],[0,1]],                 defW:20,defH:16,label:"Triangle" },
  l_shape:  {norm:[[0,0],[1,0],[1,.5],[.5,.5],[.5,1],[0,1]],defW:22,defH:18,label:"L-Shape"},
  circle:   {norm:null,                                 defW:16,defH:16,label:"Circle"   },
};
const COLORS=["#4A90D9","#E07B39","#6BBF59","#9B59B6","#16A085","#D4AC0D","#C0392B","#7F8C8D"];

function scalePoints(norm,x,y,w,h){return norm.map(([nx,ny])=>[x+nx*w,y+ny*h]);}
function shapePoints(b){
  let pts;
  if(b.shape==="circle"){
    const cx=b.x+b.w/2,cy=b.y+b.h/2,r=b.w/2;
    pts=[];
    for(let i=0;i<24;i++){const a=(2*Math.PI*i)/24;pts.push([cx+r*Math.cos(a),cy+r*Math.sin(a)]);}
  } else {
    pts=scalePoints(BUILDING_SHAPES[b.shape].norm,b.x,b.y,b.w,b.h);
  }
  if(b.r){const cx=b.x+b.w/2,cy=b.y+b.h/2;pts=pts.map(p=>rotatePt(p,cx,cy,b.r));}
  return pts;
}

// ── SVG helpers ──────────────────────────────────────────────
const NS="http://www.w3.org/2000/svg";
function el(tag,attrs){
  const e=document.createElementNS(NS,tag);
  for(const k in attrs)e.setAttribute(k,attrs[k]);
  return e;
}
function pointsToAttr(pts){return pts.map(p=>p.join(",")).join(" ");}

function makeHandles(b){
  const cx=b.x+b.w/2,cy=b.y+b.h/2;
  const group=el("g",{});
  if(b.r)group.setAttribute("transform",`rotate(${b.r},${cx},${cy})`);
  const corners=[["nw",b.x,b.y],["ne",b.x+b.w,b.y],["sw",b.x,b.y+b.h],["se",b.x+b.w,b.y+b.h]];
  for(const[corner,px,py] of corners){
    if(b.shape==="circle"&&corner!=="se")continue;
    group.appendChild(el("rect",{
      x:px-2.5,y:py-2.5,width:5,height:5,
      fill:"#fff",stroke:"#4A90D9","stroke-width":1,
      "data-role":"handle","data-target":"resize","data-id":b.id,style:`cursor:${corner}-resize`,
    }));
  }
  group.appendChild(el("line",{x1:cx,y1:b.y,x2:cx,y2:b.y-14,stroke:"#4A90D9","stroke-width":1,"stroke-dasharray":"2,2"}));
  group.appendChild(el("circle",{
    cx,cy:b.y-14,r:3.5,fill:"#4A90D9",stroke:"#fff","stroke-width":1,
    "data-role":"handle","data-target":"rotate","data-id":b.id,style:"cursor:crosshair",
  }));
  return group;
}

// ── Render ───────────────────────────────────────────────────
function render(){
  svg.innerHTML="";

  // Defs: grid pattern + obstacle hatch
  const defs=el("defs",{});
  const pattern=el("pattern",{id:"grid",width:10,height:10,patternUnits:"userSpaceOnUse"});
  pattern.appendChild(el("path",{d:"M 10 0 L 0 0 0 10",fill:"none",stroke:"#E9EFF5","stroke-width":0.5}));
  defs.appendChild(pattern);

  // Obstacle hatch pattern
  const hatch=el("pattern",{id:"obstacleHatch",width:6,height:6,patternUnits:"userSpaceOnUse",patternTransform:"rotate(45)"});
  hatch.appendChild(el("line",{x1:0,y1:0,x2:0,y2:6,stroke:"#E67E22","stroke-width":1.5,opacity:0.5}));
  defs.appendChild(hatch);
  svg.appendChild(defs);

  // Background grid
  svg.appendChild(el("rect",{x:-5000,y:-5000,width:10000,height:10000,fill:"url(#grid)","data-role":"bg"}));

  // Obstacles (hatched, drawn before boundary outline so boundary sits on top)
  for(const obsPts of (STATE.obstacles||[])){
    if(obsPts.length<3)continue;
    const g=el("g",{});
    g.appendChild(el("polygon",{
      points:pointsToAttr(obsPts),
      fill:"url(#obstacleHatch)","fill-opacity":0.7,
      stroke:"#E67E22","stroke-width":1.2,"stroke-dasharray":"4,2",
    }));
    svg.appendChild(g);
  }

  // Boundary
  const bPts=STATE.boundary.points;
  const isBndSel=selected&&selected.type==="boundary";
  svg.appendChild(el("polygon",{
    points:pointsToAttr(bPts),fill:"rgba(74,144,217,0.04)",
    stroke:isBndSel?"#4A90D9":"#9FB4CC","stroke-width":isBndSel?2:1.4,
    "stroke-dasharray":"5,3","data-role":"boundary",
  }));
  if(isBndSel){
    const vg=el("g",{});
    bPts.forEach(([x,y],i)=>{
      vg.appendChild(el("circle",{
        cx:x,cy:y,r:3.5,fill:"#fff",stroke:"#4A90D9","stroke-width":1.5,
        "data-role":"handle","data-target":"boundary-vertex","data-index":i,style:"cursor:move",
      }));
    });
    svg.appendChild(vg);
  }

  // Buildings
  STATE.buildings.forEach((b,idx)=>{
    const pts=shapePoints(b);
    const inside=pts.every(p=>pointInPolygon(p[0],p[1],bPts));
    const overlapping=STATE.buildings.some((o,j)=>j!==idx&&polysOverlap(pts,shapePoints(o)));
    const isSel=selected&&selected.type==="building"&&selected.id===b.id;

    const g=el("g",{"data-role":"building","data-id":b.id,style:"cursor:move"});
    g.appendChild(el("polygon",{
      points:pointsToAttr(pts),
      fill:b.color,"fill-opacity":isSel?0.9:0.65,
      stroke:(!inside||overlapping)?"#E74C3C":b.color,
      "stroke-width":isSel?2.5:1.5,
      "stroke-dasharray":(!inside||overlapping)?"4,2":"none",
    }));
    const cx=b.x+b.w/2,cy=b.y+b.h/2;
    const bboxH=getBBox(pts);
    const fsize=Math.max(1.8,Math.min(5,Math.min(b.w,b.h)*0.28));
    const label=el("text",{x:cx,y:cy,"text-anchor":"middle","dominant-baseline":"middle",
      "font-size":fsize,fill:"#1F2D3D","pointer-events":"none","font-weight":"600"});
    if(b.r)label.setAttribute("transform",`rotate(${b.r},${cx},${cy})`);
    label.textContent=b.name;
    g.appendChild(label);
    if(!inside||overlapping){
      const warn=el("text",{x:bboxH.minX,y:bboxH.minY-2,"font-size":7,"pointer-events":"none"});
      warn.textContent="⚠️"; g.appendChild(warn);
    }
    svg.appendChild(g);
    if(isSel)svg.appendChild(makeHandles(b));
  });

  renderSidePanel();
}

// ── Side panel ───────────────────────────────────────────────
function renderSidePanel(){
  const siteArea=shoelaceArea(STATE.boundary.points);
  const builtArea=STATE.buildings.reduce((s,b)=>s+(b.shape==="circle"?Math.PI*(b.w/2)*(b.w/2):shoelaceArea(shapePoints(b))),0);
  statsRowEl.innerHTML=`
    <div class="stat">Site area<b>${siteArea.toFixed(0)} m&sup2;</b></div>
    <div class="stat">Built area<b>${builtArea.toFixed(0)} m&sup2;</b></div>
    <div class="stat">Utilisation<b>${siteArea>0?((builtArea/siteArea)*100).toFixed(1):0}%</b></div>
    <div class="stat">Buildings<b>${STATE.buildings.length}</b></div>
  `;

  bldListEl.innerHTML="";
  if(STATE.obstacles&&STATE.obstacles.length>0){
    bldListEl.innerHTML+=`<div class="obstacle-badge">🚧 ${STATE.obstacles.length} obstacle zone${STATE.obstacles.length>1?"s":""} (from CAD)</div>`;
  }
  if(STATE.buildings.length===0){
    bldListEl.innerHTML+='<div class="empty-list">No buildings yet.<br>Add shapes above or use Auto-Arrange.</div>';
    return;
  }

  STATE.buildings.forEach(b=>{
    const pts=shapePoints(b);
    const inside=pts.every(p=>pointInPolygon(p[0],p[1],STATE.boundary.points));
    const row=document.createElement("div");
    row.className="bld-row"+(selected&&selected.type==="building"&&selected.id===b.id?" selected":"");
    row.addEventListener("pointerdown",()=>{
      STATE.buildings=STATE.buildings.filter(x=>x.id!==b.id); STATE.buildings.push(b);
      selected={type:"building",id:b.id}; render();
    });
    const top=document.createElement("div"); top.className="top";
    const cp=document.createElement("input"); cp.type="color"; cp.className="color-picker"; cp.value=b.color;
    cp.addEventListener("input",e=>{b.color=e.target.value;render();});
    cp.addEventListener("change",()=>syncToPython());
    const ni=document.createElement("input"); ni.type="text"; ni.value=b.name;
    ni.addEventListener("input",e=>b.name=e.target.value);
    ni.addEventListener("change",()=>{render();syncToPython();});
    const dupBtn=document.createElement("button"); dupBtn.className="icon-btn"; dupBtn.title="Duplicate"; dupBtn.innerHTML="⧉";
    dupBtn.addEventListener("click",e=>{
      e.stopPropagation();
      const clone=JSON.parse(JSON.stringify(b));
      clone.id=nextId++; clone.x+=5; clone.y+=5; clone.name+=" (Copy)";
      STATE.buildings.push(clone); selected={type:"building",id:clone.id};
      render(); syncToPython();
    });
    const delBtn=document.createElement("button"); delBtn.className="icon-btn del"; delBtn.title="Delete"; delBtn.innerHTML="&#10005;";
    delBtn.addEventListener("click",e=>{
      e.stopPropagation(); STATE.buildings=STATE.buildings.filter(x=>x.id!==b.id);
      if(selected&&selected.id===b.id)selected=null;
      render(); syncToPython();
    });
    top.appendChild(cp); top.appendChild(ni); top.appendChild(dupBtn); top.appendChild(delBtn);
    const meta=document.createElement("div"); meta.className="meta";
    meta.innerHTML=`<b>${BUILDING_SHAPES[b.shape].label}</b>`+(inside?"":'&nbsp;·&nbsp;<span class="warn-badge">outside boundary</span>');
    const dimsRow=document.createElement("div"); dimsRow.className="dims-row";
    let dimHTML=b.shape==="circle"
      ?`<span>Ø:</span><input type="number" data-prop="w" value="${Math.round(b.w)}" min="1" step="1"><span class="unit">m</span>`
      :`<span>W:</span><input type="number" data-prop="w" value="${Math.round(b.w)}" min="1" step="1">&nbsp;<span>H:</span><input type="number" data-prop="h" value="${Math.round(b.h)}" min="1" step="1"><span class="unit">m</span>`;
    dimHTML+=`<span style="margin-left:6px">R:</span><input type="number" data-prop="r" value="${b.r||0}" step="1" style="width:40px"><span class="unit">&deg;</span>`;
    dimsRow.innerHTML=dimHTML;
    dimsRow.querySelectorAll("input").forEach(inp=>{
      inp.addEventListener("change",e=>{
        const prop=e.target.getAttribute("data-prop");
        let val=Number(e.target.value);
        if(prop==="r"){b.r=(val%360+360)%360;}
        else{val=Math.max(1,val);b[prop]=val;if(b.shape==="circle"&&prop==="w")b.h=val;}
        render(); syncToPython();
      });
    });
    row.appendChild(top); row.appendChild(meta); row.appendChild(dimsRow);
    bldListEl.appendChild(row);
  });
}

// ── syncToPython ─────────────────────────────────────────────
function syncToPython(){
  const siteArea=shoelaceArea(STATE.boundary.points);
  const builtArea=STATE.buildings.reduce((s,b)=>s+(b.shape==="circle"?Math.PI*(b.w/2)*(b.w/2):shoelaceArea(shapePoints(b))),0);
  if(svg.__setStateValue){
    svg.__setStateValue("layout",{
      boundary:STATE.boundary, obstacles:STATE.obstacles||[],
      buildings:STATE.buildings,
      site_area:siteArea, built_area:builtArea,
      utilization_pct:siteArea>0?(builtArea/siteArea)*100:0,
    });
  }
}

// ── Controls ─────────────────────────────────────────────────
parentElement.querySelector("#snapGrid").addEventListener("change",e=>{SNAP=e.target.checked?5:1;});
parentElement.querySelector("#clearBtn").addEventListener("click",()=>{
  if(confirm("Clear all buildings from canvas?")){ STATE.buildings=[]; selected=null; render(); syncToPython(); }
});

svg.addEventListener("wheel",e=>{
  e.preventDefault();
  const zf=e.deltaY>0?1.08:0.92;
  const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY;
  const ctm=svg.getScreenCTM(); if(!ctm)return;
  const p=pt.matrixTransform(ctm.inverse());
  const targetW=clamp(viewW*zf,1,5000);
  const targetH=targetW*(viewH/viewW);
  viewX=p.x-(p.x-viewX)*(targetW/viewW);
  viewY=p.y-(p.y-viewY)*(targetH/viewH);
  viewW=targetW; viewH=targetH; updateViewBox();
},{passive:false});

function addBuilding(shape,worldX,worldY){
  const def=BUILDING_SHAPES[shape];
  const x=worldX!==undefined?snap(worldX-def.defW/2):snap(viewX+viewW/3);
  const y=worldY!==undefined?snap(worldY-def.defH/2):snap(viewY+viewH/3);
  const b={
    id:nextId++, shape, x, y, w:def.defW, h:def.defH, r:0,
    name:`${def.label} ${STATE.buildings.length+1}`,
    color:COLORS[STATE.buildings.length%COLORS.length],
  };
  STATE.buildings.push(b); selected={type:"building",id:b.id};
  render(); syncToPython();
}

parentElement.querySelectorAll(".shape-btn[data-shape]").forEach(btn=>{
  btn.addEventListener("click",()=>addBuilding(btn.dataset.shape));
  btn.addEventListener("dragstart",e=>e.dataTransfer.setData("text",btn.dataset.shape));
});
svg.addEventListener("dragover",e=>e.preventDefault());
svg.addEventListener("drop",e=>{
  e.preventDefault(); const shape=e.dataTransfer.getData("text"); if(!BUILDING_SHAPES[shape])return;
  const[wx,wy]=screenToWorld(e.clientX,e.clientY);
  addBuilding(shape,wx,wy);
});

bSel.addEventListener("change",()=>{
  const p=bSel.value;
  if(p==="custom")return; // don't overwrite CAD-imported boundary with "custom" placeholder
  STATE.boundary={preset:p,sides:BOUNDARY_PRESETS[p].length,points:BOUNDARY_PRESETS[p].map(x=>[...x])};
  STATE.obstacles=[];
  bSides.value=STATE.boundary.sides; fitViewToState(); render(); syncToPython();
});
bSides.addEventListener("change",()=>{
  let n=clamp(Math.round(Number(bSides.value))||4,3,20); bSides.value=n;
  let sx=0,sy=0; STATE.boundary.points.forEach(p=>{sx+=p[0];sy+=p[1];});
  STATE.boundary={preset:"custom",sides:n,points:genNGon(n,sx/STATE.boundary.points.length,sy/STATE.boundary.points.length,BOUNDARY_DEFAULT_R)};
  fitViewToState(); render(); syncToPython();
});

// ── Pointer events ───────────────────────────────────────────
svg.addEventListener("pointerdown",e=>{
  const target=e.target.closest("[data-role]");
  const[wx,wy]=screenToWorld(e.clientX,e.clientY);
  if(!target||target.getAttribute("data-role")==="bg"){
    selected=null;
    drag={mode:"pan",startClientX:e.clientX,startClientY:e.clientY,startViewX:viewX,startViewY:viewY};
    svg.setPointerCapture(e.pointerId); render(); return;
  }
  const role=target.getAttribute("data-role");
  if(role==="handle"){
    const tType=target.getAttribute("data-target");
    if(tType==="boundary-vertex"){
      selected={type:"boundary"}; drag={mode:"vertex",index:Number(target.getAttribute("data-index"))};
    } else {
      const obj=STATE.buildings.find(b=>String(b.id)===target.getAttribute("data-id")); if(!obj)return;
      if(tType==="rotate")drag={mode:"rotate",obj};
      else {
        const ctm=target.closest("g")?target.closest("g").getScreenCTM():svg.getScreenCTM();
        drag={mode:"resize",obj,isCircle:obj.shape==="circle",ctm:ctm?ctm.inverse():null};
      }
    }
    svg.setPointerCapture(e.pointerId); return;
  }
  if(role==="building"){
    const id=Number(target.getAttribute("data-id")); const b=STATE.buildings.find(x=>x.id===id); if(!b)return;
    STATE.buildings=STATE.buildings.filter(x=>x.id!==id); STATE.buildings.push(b);
    selected={type:"building",id};
    drag={mode:"move",obj:b,offX:wx-b.x,offY:wy-b.y};
    svg.setPointerCapture(e.pointerId); render(); return;
  }
  if(role==="boundary"){selected={type:"boundary"};render();}
});

svg.addEventListener("pointermove",e=>{
  if(!drag)return;
  if(drag.mode==="pan"){
    const fx=viewW/svg.clientWidth, fy=viewH/svg.clientHeight;
    viewX=drag.startViewX-(e.clientX-drag.startClientX)*fx;
    viewY=drag.startViewY-(e.clientY-drag.startClientY)*fy;
    updateViewBox(); return;
  }
  const[wx,wy]=screenToWorld(e.clientX,e.clientY);
  const obj=drag.obj;
  if(drag.mode==="vertex"){
    const old=STATE.boundary.points[drag.index];
    STATE.boundary.points[drag.index]=[snap(wx),snap(wy)];
    if(shoelaceArea(STATE.boundary.points)>MAX_AREA)STATE.boundary.points[drag.index]=old;
  } else if(drag.mode==="move"){
    obj.x=snap(wx-drag.offX); obj.y=snap(wy-drag.offY);
  } else if(drag.mode==="rotate"){
    const cx=obj.x+obj.w/2,cy=obj.y+obj.h/2;
    let deg=(Math.atan2(wy-cy,wx-cx)*180/Math.PI)+90;
    deg=e.shiftKey?Math.round(deg/15)*15:Math.round(deg/5)*5;
    obj.r=(deg%360+360)%360;
  } else if(drag.mode==="resize"&&drag.ctm){
    const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY;
    const lp=pt.matrixTransform(drag.ctm);
    const cx=obj.x+obj.w/2,cy=obj.y+obj.h/2;
    if(drag.isCircle){
      const r=Math.max(MIN_SIZE/2,Math.hypot(lp.x-cx,lp.y-cy));
      obj.w=obj.h=snap(r*2);
    } else {
      obj.w=snap(Math.max(MIN_SIZE,Math.abs(lp.x-cx)*2));
      obj.h=snap(Math.max(MIN_SIZE,Math.abs(lp.y-cy)*2));
    }
    obj.x=cx-obj.w/2; obj.y=cy-obj.h/2;
  }
  render();
});

function endDrag(e){
  if(!drag)return; const bm=drag.mode; drag=null;
  try{svg.releasePointerCapture(e.pointerId);}catch(err){}
  if(bm!=="pan")syncToPython();
}
svg.addEventListener("pointerup",endDrag);
svg.addEventListener("pointercancel",endDrag);

// ── Load state ───────────────────────────────────────────────
function loadState(raw){
  let s=raw?JSON.parse(JSON.stringify(raw)):{};
  if(!s.boundary)s.boundary={preset:"rectangle",sides:4,points:BOUNDARY_PRESETS.rectangle.map(p=>[...p])};
  if(!s.obstacles)s.obstacles=[];
  if(!s.buildings)s.buildings=[];
  STATE=s;
  nextId=STATE.buildings.reduce((m,b)=>Math.max(m,b.id+1),1); selected=null;
}

loadState(data);
bSel.value=STATE.boundary.preset||"custom";
bSides.value=STATE.boundary.sides||STATE.boundary.points.length;
fitViewToState(); render();
if(!data||data.site_area===undefined)syncToPython();
}
"""

# Register the custom canvas component
_site_canvas = st.components.v2.component(
    "site_canvas_v9",
    html=_CANVAS_HTML,
    css=_CANVAS_CSS,
    js=_CANVAS_JS,
)

def site_canvas(initial_state: dict, version: int):
    return _site_canvas(
        data=initial_state,
        default={"layout": initial_state},
        key=f"canvas-v9-{version}",
        height=790,
        on_layout_change=lambda: None,
    )
# ─────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────
_DEMO_BOUNDARY = {
    "preset": "rectangle",
    "sides": 4,
    "points": [[15, 15], [215, 15], [215, 165], [15, 165]],
}
_DEFAULT_STATE = {
    "boundary": _DEMO_BOUNDARY,
    "obstacles": [],
    "buildings": [],
    "site_area": 30000,
    "built_area": 0,
    "utilization_pct": 0.0,
}

STAGE_NAMES = [s.name for s in bc.PROCESS_STAGES]
STAGE_NAME_TO_ID = {s.name: s.id for s in bc.PROCESS_STAGES}
STAGE_ID_TO_STAGE = {s.id: s for s in bc.PROCESS_STAGES}
_CUSTOM_STAGE_LABEL = "Custom / Other (not linked to flow or safety rules)"

_DEFAULT_REQUIREMENTS = pd.DataFrame([
    {
        "Stage": s.name, "Width (m)": s.default_w, "Height (m)": s.default_h,
        "Count": s.default_count, "Allow Rotate": s.allow_rotate, "Color": s.color,
    }
    for s in bc.PROCESS_STAGES
])

_DEFAULT_FLOW_DF = pd.DataFrame([
    {"From": STAGE_ID_TO_STAGE[a].name, "To": STAGE_ID_TO_STAGE[b].name, "Weight": w}
    for a, b, w in bc.DEFAULT_FLOW_EDGES
])

_DEFAULT_SAFETY_DF = pd.DataFrame([
    {"From": STAGE_ID_TO_STAGE[a].name, "To": STAGE_ID_TO_STAGE[b].name,
     "Min Distance (m)": d, "Reason": reason}
    for a, b, d, reason in bc.DEFAULT_SAFETY_RULES
])


def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("site_state", copy.deepcopy(_DEFAULT_STATE))
_init("version", 1)
_init("last_upload_id", None)
_init("last_dxf_id", None)
_init("layout_options", None)      # last list[LayoutResult] from generate_layouts()
_init("active_layout_idx", 0)
_init("opt_error", None)
_init("pdf_raw_result", None)      # uncalibrated DxfImportResult from pdf_import, pending scale input
_init("pdf_info", None)            # PdfImportInfo (extraction_method, raw_width/height, page info)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _state_boundary_polygon() -> Polygon:
    pts = st.session_state.site_state["boundary"]["points"]
    return Polygon([(p[0], p[1]) for p in pts])

def _state_obstacle_polygons() -> list:
    raw = st.session_state.site_state.get("obstacles", [])
    polys = []
    for pts in raw:
        if len(pts) >= 3:
            polys.append(Polygon([(p[0], p[1]) for p in pts]))
    return polys

def _poly_to_points(poly: Polygon) -> list:
    return [[x, y] for x, y in poly.exterior.coords[:-1]]

def _apply_cad_import_result(result, source_label: str):
    """Shared by both the DXF and PDF import paths — both return the same
    dxf_import.DxfImportResult type, so equipment classification, obstacle
    handling, and canvas state population are identical either way.
    """
    classified = [z for z in result.equipment_zones if z.stage_id]
    unclassified = [z for z in result.equipment_zones if not z.stage_id]

    existing_buildings = []
    for i, z in enumerate(classified):
        minx, miny, maxx, maxy = z.polygon.bounds
        st_obj = STAGE_ID_TO_STAGE.get(z.stage_id)
        existing_buildings.append({
            "id": i + 1, "shape": "rectangle",
            "x": round(minx, 4), "y": round(miny, 4),
            "w": round(maxx - minx, 4), "h": round(maxy - miny, 4),
            "r": 0,
            "name": f"Existing: {z.stage_name}",
            "color": st_obj.color if st_obj else "#888888",
            "stage_id": z.stage_id,
            "existing": True,
        })

    new_state = copy.deepcopy(st.session_state.site_state)
    new_state["boundary"] = {
        "preset": "custom",
        "sides": len(result.boundary.exterior.coords) - 1,
        "points": _poly_to_points(result.boundary),
    }
    new_state["obstacles"] = [_poly_to_points(z.polygon) for z in unclassified]
    new_state["buildings"] = existing_buildings
    new_state["site_area"] = result.boundary.area
    built_area = sum(b["w"] * b["h"] for b in existing_buildings)
    new_state["built_area"] = built_area
    new_state["utilization_pct"] = (built_area / result.boundary.area * 100) if result.boundary.area > 0 else 0.0

    st.session_state.site_state = new_state
    st.session_state.version += 1
    st.session_state.layout_options = None
    st.session_state.opt_error = None

    st.success(
        f"✅ Imported from {source_label}: {result.boundary.area:,.0f} m²  ·  "
        f"{len(classified)} classified equipment  ·  "
        f"{len(unclassified)} unclassified obstacle(s)  ·  "
        f"units: {result.detected_unit}"
    )
    for w in result.warnings:
        st.info(f"ℹ️ {w}")

def _placements_to_buildings(placements, start_id: int = 1) -> list:
    buildings = []
    for i, p in enumerate(placements):
        buildings.append({
            "id": start_id + i,
            "shape": "rectangle",
            "x": round(p.x, 4),
            "y": round(p.y, 4),
            "w": round(p.w, 4),
            "h": round(p.h, 4),
            "r": 0,
            "name": p.name,
            "color": p.color,
            "stage_id": p.stage_id,
            "existing": False,
        })
    return buildings

def _rows_to_specs(df: pd.DataFrame) -> list:
    specs = []
    seen_stage_ids = {}
    for _, row in df.iterrows():
        stage_label = str(row.get("Stage", "")).strip()
        if not stage_label:
            continue
        try:
            w = float(row["Width (m)"])
            h = float(row["Height (m)"])
            cnt = int(row["Count"])
            rotate = bool(row.get("Allow Rotate", True))
            color = str(row.get("Color", "#4A90D9"))
        except (ValueError, KeyError, TypeError):
            continue
        if w <= 0 or h <= 0 or cnt <= 0:
            continue

        flow_stage_id = STAGE_NAME_TO_ID.get(stage_label)  # None for custom rows
        if flow_stage_id is not None:
            n = seen_stage_ids.get(flow_stage_id, 0)
            seen_stage_ids[flow_stage_id] = n + 1
            unique_id = flow_stage_id if n == 0 else f"{flow_stage_id}__{n}"
        else:
            unique_id = str(uuid.uuid4())[:8]

        specs.append(BuildingSpec(
            id=unique_id, name=stage_label, width=w, height=h,
            count=cnt, allow_rotate=rotate, color=color,
            flow_stage_id=flow_stage_id,
        ))
    return specs

def _rows_to_flow_edges(df: pd.DataFrame) -> list:
    edges = []
    for _, row in df.iterrows():
        a = STAGE_NAME_TO_ID.get(str(row.get("From", "")).strip())
        b = STAGE_NAME_TO_ID.get(str(row.get("To", "")).strip())
        try:
            w = float(row["Weight"])
        except (ValueError, KeyError, TypeError):
            continue
        if a and b and a != b and w > 0:
            edges.append(FlowEdge(from_id=a, to_id=b, weight=w))
    return edges

def _rows_to_safety_rules(df: pd.DataFrame) -> list:
    rules = []
    for _, row in df.iterrows():
        a = STAGE_NAME_TO_ID.get(str(row.get("From", "")).strip())
        b = STAGE_NAME_TO_ID.get(str(row.get("To", "")).strip())
        try:
            d = float(row["Min Distance (m)"])
        except (ValueError, KeyError, TypeError):
            continue
        reason = str(row.get("Reason", "") or "")
        if a and b and a != b and d > 0:
            rules.append(SafetyRule(from_id=a, to_id=b, min_distance=d, reason=reason))
    return rules

def _current_locked_placements() -> list:
    placements = []
    for b in st.session_state.site_state.get("buildings", []):
        placements.append(Placement(
            spec_id=f"locked_{b['id']}", instance=0,
            name=b.get("name", ""), color=b.get("color", "#999"),
            x=b["x"], y=b["y"], w=b["w"], h=b["h"],
            stage_id=b.get("stage_id"),
        ))
    return placements

def _rect_gap(a: Placement, b: Placement) -> float:
    """Axis-aligned separation gap actually achieved between two placed
    rectangles — a lower-bound estimate of true clearance (the hard
    constraint guarantees at least one axis has this gap >= the required
    minimum, which alone guarantees real-world straight-line distance >=
    that minimum too).
    """
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    x_gap = max(bx0 - ax1, ax0 - bx1, 0.0)
    y_gap = max(by0 - ay1, ay0 - by1, 0.0)
    return max(x_gap, y_gap)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛽ CBG Plant Layout Planner")
    st.caption("CAD Import · Process-Flow Optimizer · v10.0")
    st.divider()

    # ── Section 1: Import ────────────────────────────────────
    with st.expander("📐 Import Plant CAD or Plan (DXF/PDF)", expanded=True):
        dxf_file = st.file_uploader(
            "Upload plant DXF or PDF file",
            type=["dxf", "pdf"],
            help="DXF: the outermost closed shape becomes the site boundary; interior shapes "
                 "auto-classify against known biogas/CBG equipment names. PDF: same idea, but "
                 "read via vector-content extraction (if the PDF has real CAD vector data) or "
                 "OCR + shape detection (if it's a scan) — you'll be asked for a real-world "
                 "dimension afterward since PDF has no embedded scale like DXF's $INSUNITS does.",
        )
        col_layer, col_scale = st.columns(2)
        dxf_layer = col_layer.text_input(
            "Boundary layer/text hint",
            placeholder="e.g. SITE",
            help="Leave blank for auto-detect (largest closed shape). For PDF this matches "
                 "against text found near/inside a shape instead of a DXF layer name.",
        )
        dxf_scale_opt = col_scale.selectbox(
            "DXF units override",
            ["Auto-detect", "mm → m", "cm → m", "m (no change)", "inches → m", "feet → m"],
            help="Only applies to DXF uploads — PDF is calibrated separately below after import.",
        )
        _scale_map = {
            "Auto-detect": None, "mm → m": 0.001, "cm → m": 0.01,
            "m (no change)": 1.0, "inches → m": 0.0254, "feet → m": 0.3048,
        }
        manual_scale = _scale_map[dxf_scale_opt]

        if dxf_file is not None:
            dxf_id = (dxf_file.name, dxf_file.size)
            is_pdf = dxf_file.name.lower().endswith(".pdf")
            if dxf_id != st.session_state.last_dxf_id:
                if is_pdf:
                    with st.spinner("Extracting from PDF (OCR can take a few seconds)…"):
                        try:
                            raw_result, pdf_info = pdf_import.import_pdf(
                                dxf_file.read(),
                                boundary_hint=dxf_layer.strip() or None,
                            )
                            st.session_state.pdf_raw_result = raw_result
                            st.session_state.pdf_info = pdf_info
                            st.session_state.last_dxf_id = dxf_id
                        except pdf_import.PdfImportError as e:
                            st.error(f"Import failed: {e}")
                else:
                    with st.spinner("Parsing DXF…"):
                        try:
                            result = import_dxf(
                                dxf_file.read(),
                                boundary_layer=dxf_layer.strip() or None,
                                manual_unit_scale=manual_scale,
                            )
                            st.session_state.pdf_raw_result = None
                            st.session_state.last_dxf_id = dxf_id
                            _apply_cad_import_result(result, "DXF")
                        except DxfImportError as e:
                            st.error(f"Import failed: {e}")

        # PDF calibration step — shown whenever there's an extracted-but-
        # not-yet-calibrated PDF result waiting (persists across reruns
        # until "Apply calibration" is clicked, so adjusting the number
        # doesn't require re-running OCR/vector extraction each time).
        if st.session_state.pdf_raw_result is not None:
            raw = st.session_state.pdf_raw_result
            pinfo = st.session_state.pdf_info
            minx, miny, maxx, maxy = raw.boundary.bounds
            raw_w, raw_h = maxx - minx, maxy - miny
            st.warning(
                f"Extracted via **{pinfo.extraction_method}** — raw boundary size "
                f"{raw_w:.0f} × {raw_h:.0f} pt. PDF has no embedded real-world scale "
                f"(unlike DXF) — calibrate before optimizing:",
                icon="📏",
            )
            calib_mode = st.radio(
                "Calibrate using", ["Known boundary width (m)", "Known boundary height (m)", "Known plot scale (1:N)"],
                horizontal=False, key="pdf_calib_mode",
            )
            factor = None
            if calib_mode == "Known boundary width (m)":
                true_w = st.number_input("Real-world boundary width (m)", min_value=0.0, value=0.0, step=1.0)
                if true_w > 0:
                    factor = true_w / raw_w
            elif calib_mode == "Known boundary height (m)":
                true_h = st.number_input("Real-world boundary height (m)", min_value=0.0, value=0.0, step=1.0)
                if true_h > 0:
                    factor = true_h / raw_h
            else:
                plot_n = st.number_input("Plot scale denominator N (drawing is 1:N)", min_value=0.0, value=0.0, step=10.0,
                                          help="Only accurate if the PDF was exported/printed at 100% "
                                               "('actual size'), not 'fit to page'.")
                if plot_n > 0:
                    factor = plot_n * (0.0254 / 72.0)  # metres per PDF point at stated plot scale

            col_apply, col_skip = st.columns(2)
            if col_apply.button("✅ Apply calibration & load", width="stretch", disabled=factor is None):
                calibrated = pdf_import.rescale_result(raw, factor)
                _apply_cad_import_result(calibrated, "PDF")
                st.session_state.pdf_raw_result = None
                st.rerun()
            if col_skip.button("Discard this import", width="stretch"):
                # NOT clearing last_dxf_id here — the file is still sitting
                # in the uploader, so nulling it would make this same file
                # look "new" again on the rerun below and immediately
                # re-trigger extraction, undefeating the point of "discard".
                st.session_state.pdf_raw_result = None
                st.session_state.pdf_info = None
                st.rerun()

        st.divider()
        json_file = st.file_uploader("Or load saved layout (JSON)", type=["json"])
        if json_file is not None:
            jid = (json_file.name, json_file.size)
            if jid != st.session_state.last_upload_id:
                try:
                    loaded = json.loads(json_file.read())
                    if "boundary" in loaded and "buildings" in loaded:
                        st.session_state.site_state = loaded
                        st.session_state.version += 1
                        st.session_state.last_upload_id = jid
                        st.session_state.layout_options = None
                        st.success("Layout loaded.")
                except Exception:
                    st.error("Couldn't read JSON — is it a valid layout file?")

    # ── Section 2: Equipment to place ────────────────────────
    with st.expander("🏭 Process Equipment To Place", expanded=True):
        st.caption(
            "Defaults to a standard CBG plant catalog. Set Count = 0 to exclude a stage "
            "without deleting the row; add rows for anything site-specific."
        )
        req_df = st.data_editor(
            _DEFAULT_REQUIREMENTS,
            column_config={
                "Stage": st.column_config.SelectboxColumn(
                    "Stage", options=STAGE_NAMES + [_CUSTOM_STAGE_LABEL], width="medium"),
                "Width (m)":    st.column_config.NumberColumn("Width (m)",  min_value=0.5, step=0.5, format="%.1f"),
                "Height (m)":   st.column_config.NumberColumn("Height (m)", min_value=0.5, step=0.5, format="%.1f"),
                "Count":        st.column_config.NumberColumn("Count", min_value=0, max_value=50, step=1),
                "Allow Rotate": st.column_config.CheckboxColumn("Rotate?"),
                "Color":        st.column_config.TextColumn("Color hex", width="small"),
            },
            num_rows="dynamic",
            width='stretch',
            key="req_table",
        )

    # ── Section 3: Process flow ──────────────────────────────
    with st.expander("🔀 Process Flow (drives the optimizer)", expanded=False):
        st.caption(
            "Higher weight = keeping this connection short matters more (raw/upgraded gas "
            "transfer lines are far more distance-sensitive than, e.g., security-to-gate)."
        )
        flow_df = st.data_editor(
            _DEFAULT_FLOW_DF,
            column_config={
                "From":   st.column_config.SelectboxColumn("From", options=STAGE_NAMES, width="medium"),
                "To":     st.column_config.SelectboxColumn("To", options=STAGE_NAMES, width="medium"),
                "Weight": st.column_config.NumberColumn("Weight", min_value=0.1, max_value=20.0, step=0.5),
            },
            num_rows="dynamic",
            width='stretch',
            key="flow_table",
        )

    # ── Section 4: Safety clearances ─────────────────────────
    with st.expander("🚧 Safety Clearances (hard constraints)", expanded=False):
        st.warning(
            "These are illustrative engineering-judgment defaults, **not verified regulatory "
            "values**. Verify against the codes that apply in your jurisdiction (e.g. OISD/PESO "
            "in India, NFPA 30/58 elsewhere) and have the layout reviewed by a qualified process "
            "safety engineer before construction. A generated layout will never place two linked "
            "stages closer than the distance you set here.",
            icon="⚠️",
        )
        safety_df = st.data_editor(
            _DEFAULT_SAFETY_DF,
            column_config={
                "From":              st.column_config.SelectboxColumn("From", options=STAGE_NAMES, width="medium"),
                "To":                st.column_config.SelectboxColumn("To", options=STAGE_NAMES, width="medium"),
                "Min Distance (m)":  st.column_config.NumberColumn("Min Distance (m)", min_value=0.0, step=1.0),
                "Reason":            st.column_config.TextColumn("Reason", width="large"),
            },
            num_rows="dynamic",
            width='stretch',
            key="safety_table",
        )

    # ── Section 5: Solver settings ───────────────────────────
    with st.expander("⚙️ Optimizer Settings", expanded=False):
        cell_size = st.select_slider(
            "Grid cell size",
            options=[0.25, 0.5, 1.0, 2.0],
            value=1.0,
            format_func=lambda v: f"{v} m",
            help="Smaller cells = finer placement but slower. 1 m is good for most sites.",
        )
        col_s1, col_s2 = st.columns(2)
        setback = col_s1.number_input(
            "Boundary setback (m)",
            min_value=0.0, max_value=20.0, value=3.0, step=0.5,
            help="Minimum clearance between any equipment and the site boundary/obstacles.",
        )
        spacing = col_s2.number_input(
            "Baseline equipment spacing (m)",
            min_value=0.0, max_value=20.0, value=2.0, step=0.5,
            help="Minimum gap between any two equipment items in general — the safety "
                 "clearances above apply on top of this for specific hazardous pairs.",
        )
        time_limit = st.slider(
            "Solver time limit per layout (s)",
            min_value=5, max_value=120, value=10, step=5,
            help="Applied to Phase 1 (fit-maximization) and, in parallel, to each of the 3 "
                 "layout strategies — so total run time is roughly 2x this value, not 4x. "
                 "Most layouts converge well before the limit; this is a ceiling, not a target.",
        )
        keep_existing = st.checkbox(
            "Lock current canvas equipment in place",
            value=True,
            help="Treats equipment already on the canvas (including anything imported from the "
                 "DXF) as fixed, and arranges new equipment around it.",
        )

    # ── Section 6: Run ───────────────────────────────────────
    st.divider()
    run_btn = st.button(
        "🧬 Generate Layouts",
        width='stretch',
        type="primary",
        help="Runs the CP-SAT process-flow optimizer and produces 3 alternative layouts.",
    )
    if st.button("🔄 Reset to blank layout", width='stretch'):
        # Deliberately NOT clearing last_dxf_id/last_upload_id: the file
        # uploader widgets keep whatever file was last selected across
        # reruns, so nulling those ids here would make the exact same file
        # look "new" again on the very next rerun (this button triggers one
        # via st.rerun() below) and silently re-import it, undoing this
        # reset. Clearing them is safe only once the person actually
        # removes/changes the uploaded file themselves.
        st.session_state.site_state = copy.deepcopy(_DEFAULT_STATE)
        st.session_state.version += 1
        st.session_state.layout_options = None
        st.session_state.opt_error = None
        st.session_state.pdf_raw_result = None
        st.session_state.pdf_info = None
        st.rerun()

    st.divider()
    st.download_button(
        "💾 Download layout (JSON)",
        data=json.dumps(st.session_state.site_state, indent=2),
        file_name="cbg_plant_layout.json",
        mime="application/json",
        width='stretch',
    )


# ─────────────────────────────────────────────────────────────
# GENERATE LAYOUTS
# ─────────────────────────────────────────────────────────────
if run_btn:
    specs = _rows_to_specs(req_df)
    if not specs:
        st.session_state.opt_error = (
            "No valid equipment rows — set at least one row's Count ≥ 1 with a Stage, "
            "positive Width and Height."
        )
        st.session_state.layout_options = None
    else:
        flow_edges = _rows_to_flow_edges(flow_df)
        safety_rules = _rows_to_safety_rules(safety_df)
        boundary_poly = _state_boundary_polygon()
        obstacle_polys = _state_obstacle_polygons()
        locked_placements = _current_locked_placements() if keep_existing else []

        progress = st.sidebar.progress(0, text="Rasterising site…")
        try:
            grid = rasterize(
                boundary=boundary_poly,
                obstacles=obstacle_polys,
                cell_size=cell_size,
                setback=setback,
            )
            progress.progress(15, text="Solving Phase 1 (maximize fit)…")
            layouts = generate_layouts(
                grid=grid,
                specs=specs,
                flow_edges=flow_edges,
                safety_rules=safety_rules,
                spacing=spacing,
                time_limit_s=time_limit,
                locked=locked_placements if locked_placements else None,
            )
            progress.progress(95, text="Applying best layout…")

            st.session_state.layout_options = layouts
            st.session_state.active_layout_idx = 0
            st.session_state.opt_error = None

            # Apply the first strategy to the canvas by default.
            chosen = layouts[0]
            all_buildings = []
            if keep_existing:
                all_buildings = copy.deepcopy(st.session_state.site_state.get("buildings", []))
                next_id = max((b["id"] for b in all_buildings), default=0) + 1
            else:
                next_id = 1
            all_buildings.extend(_placements_to_buildings(chosen.placements, start_id=next_id))

            new_state = copy.deepcopy(st.session_state.site_state)
            new_state["buildings"] = all_buildings
            built_area = sum(b["w"] * b["h"] for b in all_buildings)
            site_area = boundary_poly.area
            new_state["built_area"] = built_area
            new_state["utilization_pct"] = (built_area / site_area * 100) if site_area > 0 else 0.0
            st.session_state.site_state = new_state
            st.session_state.version += 1
            progress.progress(100, text="Done!")
        except Exception as exc:
            st.session_state.opt_error = str(exc)
            st.session_state.layout_options = None
        finally:
            try:
                progress.empty()
            except Exception:
                pass


def _apply_layout(idx: int, keep_existing_now: bool):
    layouts = st.session_state.layout_options
    if not layouts or idx >= len(layouts):
        return
    chosen = layouts[idx]
    boundary_poly = _state_boundary_polygon()

    all_buildings = []
    if keep_existing_now:
        # keep only the ones that were locked (existing / previously-fixed) —
        # i.e. strip any buildings from a *different* previously-applied
        # strategy, then re-add this strategy's placements.
        for b in st.session_state.site_state.get("buildings", []):
            if b.get("existing"):
                all_buildings.append(copy.deepcopy(b))
        next_id = max((b["id"] for b in all_buildings), default=0) + 1
    else:
        next_id = 1
    all_buildings.extend(_placements_to_buildings(chosen.placements, start_id=next_id))

    new_state = copy.deepcopy(st.session_state.site_state)
    new_state["buildings"] = all_buildings
    built_area = sum(b["w"] * b["h"] for b in all_buildings)
    site_area = boundary_poly.area
    new_state["built_area"] = built_area
    new_state["utilization_pct"] = (built_area / site_area * 100) if site_area > 0 else 0.0
    st.session_state.site_state = new_state
    st.session_state.active_layout_idx = idx
    st.session_state.version += 1


# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
st.markdown("## ⛽ CBG Plant Layout")

if st.session_state.opt_error:
    st.error(f"Optimizer error: {st.session_state.opt_error}")

layouts = st.session_state.layout_options
if layouts and layouts[0].name != "Infeasible":
    st.markdown("#### Layout alternatives")
    cols = st.columns(len(layouts))
    for i, (col, lr) in enumerate(zip(cols, layouts)):
        with col:
            active = i == st.session_state.active_layout_idx
            st.markdown(f"{'**➤ ' if active else ''}{lr.name}{'**' if active else ''}")
            st.caption(lr.description)
            if lr.metrics:
                st.caption(
                    f"Flow dist: **{lr.metrics.get('total_flow_distance_m', 0):,.0f} m**  ·  "
                    f"Longest run: **{lr.metrics.get('longest_single_run_m', 0):,.0f} m**  ·  "
                    f"Footprint: **{lr.metrics.get('footprint_extent_m', 0):,.0f} m**"
                )
            if st.button("Use this layout", key=f"use_layout_{i}", width='stretch',
                          disabled=active, type="secondary"):
                _apply_layout(i, keep_existing_now=True)
                st.rerun()
elif layouts and layouts[0].name == "Infeasible":
    st.error(
        "No feasible layout found — the equipment doesn't fit even before considering flow or "
        "compactness. Try: reducing count/size of equipment, reducing setback/spacing, using a "
        "smaller grid cell, or increasing the solver time limit."
    )

# Render canvas
canvas_result = site_canvas(
    initial_state=st.session_state.site_state,
    version=st.session_state.version,
)
if canvas_result is not None and canvas_result.get("layout") is not None:
    st.session_state.site_state = canvas_result["layout"]

# ── Metrics row ──────────────────────────────────────────────
state = st.session_state.site_state
site_area = state.get("site_area", 0)
built_area = state.get("built_area", 0)
util_pct = state.get("utilization_pct", 0)
n_buildings = len(state.get("buildings", []))
n_obstacles = len(state.get("obstacles", []))

active_metrics = {}
if layouts and st.session_state.active_layout_idx < len(layouts):
    active_metrics = layouts[st.session_state.active_layout_idx].metrics or {}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Site area", f"{site_area:,.0f} m²")
m2.metric("Built area", f"{built_area:,.0f} m²")
m3.metric("Utilisation", f"{util_pct:.1f}%")
m4.metric("Equipment items", str(n_buildings))
m5.metric("Total flow distance", f"{active_metrics.get('total_flow_distance_m', 0):,.0f} m" if active_metrics else "—")
m6.metric("Longest single run", f"{active_metrics.get('longest_single_run_m', 0):,.0f} m" if active_metrics else "—")

# ── Result detail for the active layout ───────────────────────
if layouts and layouts[0].name != "Infeasible":
    active = layouts[st.session_state.active_layout_idx]
    with st.expander("📊 Active Layout Details", expanded=True):
        placed_count = len(active.placements)
        unplaced_total = sum(n for _, n in active.unplaced)
        total_requested = placed_count + unplaced_total
        c1, c2, c3 = st.columns(3)
        c1.metric("Solver status", active.status)
        c2.metric("Equipment placed", f"{placed_count} / {total_requested}")
        if unplaced_total > 0:
            c3.metric("Could not fit", str(unplaced_total), delta=f"-{unplaced_total}", delta_color="inverse")
        else:
            c3.metric("Could not fit", "0 ✅")

        if active.unplaced:
            st.warning(
                "Some equipment couldn't fit under the current safety clearances and setbacks. "
                "Try: reducing count/size, increasing the time limit, using a smaller grid cell, "
                "or relaxing spacing/setback/safety distances.",
                icon="⚠️",
            )
            rows = [{"Stage ID": sid, "Not placed": n} for sid, n in active.unplaced]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        if active.placements:
            st.markdown("**Placed equipment:**")
            placed_rows = [{
                "Name": p.name, "X (m)": round(p.x, 1), "Y (m)": round(p.y, 1),
                "W (m)": round(p.w, 1), "H (m)": round(p.h, 1), "Area (m²)": round(p.w * p.h, 1),
            } for p in active.placements]
            st.dataframe(pd.DataFrame(placed_rows), width='stretch', hide_index=True)

        # Safety clearance audit — should always pass since these are hard
        # constraints; shown for transparency/traceability.
        safety_rules_active = _rows_to_safety_rules(safety_df) if run_btn or st.session_state.layout_options else []
        if safety_rules_active and active.placements:
            by_stage = {}
            for p in active.placements:
                by_stage.setdefault(p.stage_id or p.spec_id, []).append(p)
            audit_rows = []
            for rule in safety_rules_active:
                pairs = [(a, b) for a in by_stage.get(rule.from_id, []) for b in by_stage.get(rule.to_id, []) if a is not b]
                if not pairs:
                    continue
                min_gap = min(_rect_gap(a, b) for a, b in pairs)
                audit_rows.append({
                    "From → To": f"{STAGE_ID_TO_STAGE[rule.from_id].name} → {STAGE_ID_TO_STAGE[rule.to_id].name}",
                    "Required (m)": rule.min_distance,
                    "Achieved (m)": round(min_gap, 1),
                    "OK": "✅" if min_gap >= rule.min_distance - 1e-6 else "❌",
                })
            if audit_rows:
                st.markdown("**Safety clearance audit** (hard constraints — should always show ✅):")
                st.dataframe(pd.DataFrame(audit_rows), width='stretch', hide_index=True)
