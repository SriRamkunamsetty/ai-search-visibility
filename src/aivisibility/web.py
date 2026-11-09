"""Read-only dashboard server (stdlib only): JSON API + single-page UI.
Security posture: binds 127.0.0.1 by default, no auth (put it behind your own auth proxy before exposing it),
CSP + nosniff headers, request bodies capped, all user-derived text inserted with textContent (no innerHTML)."""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from .ai_visibility import answer_details, audit_diff, get_audit_run, list_audit_runs, visibility_report

MAX_BODY = 20_000
HEADERS = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
           "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:"}


def _json(obj, status=200):
    return status, "application/json", json.dumps(obj, default=str).encode()


SIMULATED = "simulated-demo"

class App:
    def __init__(self, conn):
        self.conn, self.lock = conn, threading.Lock()

    def meta(self):
        models = [r["model"] for r in self.conn.execute("SELECT DISTINCT model FROM ai_answers ORDER BY model")]
        return {"models": models, "simulated": SIMULATED in models}

    def _model(self, q):
        models = self.meta()["models"]
        m = q.get("model", [models[0] if models else ""])[0]
        return m if m in models else None

    def route(self, method, raw_path, body):
        if method != "GET":
            return _json({"error": "method not allowed"}, 405)
        u = urlsplit(raw_path)
        p, q = u.path, parse_qs(u.query)
        with self.lock:
            if p == "/":
                return 200, "text/html; charset=utf-8", PAGE.encode()
            if p == "/healthz":
                return _json({"ok": True})
            if p == "/api/meta":
                return _json(self.meta())
            if p == "/api/ai":
                m = self._model(q)
                return _json(visibility_report(self.conn, m) if m else {"model": q.get("model", [""])[0], "titles": {}})
            if p == "/api/audits":
                return _json(list_audit_runs(self.conn))
            ma = re.fullmatch(r"/api/audits/(\d+)", p)
            if ma:
                run = get_audit_run(self.conn, int(ma[1]))
                return _json(run) if run else _json({"error": "not found"}, 404)
            if p == "/api/audits/diff":
                ra = q.get("a", [""])[0]
                rb = q.get("b", [""])[0]
                if not (ra.isdigit() and rb.isdigit()):
                    return _json({"error": "missing or invalid a and b run id parameters"}, 400)
                try:
                    return _json(audit_diff(self.conn, int(ra), int(rb)))
                except ValueError as err:
                    return _json({"error": str(err)}, 404)
            mm = re.fullmatch(r"/api/answers/(\d+)", p)
            if mm:
                m = self._model(q)
                d = answer_details(self.conn, int(mm[1]), m) if m else None
                return _json(d) if d else _json({"error": "not found"}, 404)
        return _json({"error": "not found"}, 404)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _run(self, method):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    status, ctype, body = _json({"error": "payload too large"}, 413)
                else:
                    status, ctype, body = app.route(method, self.path, self.rfile.read(n) if n else b"")
            except Exception as exc:  # never leak a stack trace to the client
                print(f"error on {self.path}: {exc!r}", file=sys.stderr)
                status, ctype, body = _json({"error": "internal error"}, 500)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._run("GET")

        def do_POST(self):
            self._run("POST")

        def log_message(self, fmt_, *args):
            pass
    return Handler


def serve(conn, host="127.0.0.1", port=8000):
    srv = ThreadingHTTPServer((host, port), make_handler(App(conn)))
    print(f"dashboard on http://{host}:{port}  (read-only, no auth)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI-Search Visibility</title><style>
:root{--paper:#f2ead8;--panel:#ebe2cb;--ink:#1c1a16;--muted:#6b6455;--rule:#cfc4a8;--accent:#c2410c;--cool:#1f4e5f;--good:#2f6b3a;--bad:#a4261b;
--serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif;--mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--paper:#15130f;--panel:#1d1a14;--ink:#ece3cc;--muted:#a39a84;--rule:#3a3528;--accent:#f08a4b;--cool:#7fc0d4;--good:#7fc98a;--bad:#f08478}}
*{box-sizing:border-box}html,body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 var(--serif)}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.banner{background:var(--accent);color:var(--paper);font:600 12px var(--mono);letter-spacing:.08em;text-transform:uppercase;padding:6px 16px;text-align:center}
.wrap{display:grid;grid-template-columns:270px 1fr;min-height:100vh}
aside{border-right:1px solid var(--rule);padding:22px 0}
aside h2{font:600 11px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 20px 10px}
.slate button{all:unset;display:flex;justify-content:space-between;gap:8px;width:100%;box-sizing:border-box;padding:9px 20px;cursor:pointer;border-left:3px solid transparent;border-bottom:1px solid var(--rule)}
.slate button:hover{background:var(--panel)}.slate button[aria-current=true]{border-left-color:var(--accent);background:var(--panel)}
.slate small{display:block;color:var(--muted);font:11px var(--mono);text-transform:uppercase;letter-spacing:.08em}
main{padding:26px 34px 60px;min-width:0}
.kicker{font:600 11px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
h1{font:700 clamp(30px,4.4vw,52px)/1.02 var(--serif);letter-spacing:-.02em;margin:6px 0 18px}
.figs{display:grid;grid-template-columns:repeat(4,1fr);border-top:2px solid var(--ink);border-bottom:1px solid var(--rule);margin-bottom:22px}
.fig{padding:12px 16px;border-right:1px solid var(--rule)}.fig:last-child{border-right:0}
.fig b{display:block;font:600 clamp(20px,2.6vw,32px) var(--mono)}.fig span{font:11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
nav.tabs{display:flex;gap:22px;border-bottom:1px solid var(--rule);margin-bottom:22px}
nav.tabs button{all:unset;cursor:pointer;padding:8px 0;font:600 12px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);border-bottom:2px solid transparent}
nav.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--accent)}
.read{font-size:19px;max-width:64ch;margin:0 0 10px}
.badge{display:inline-block;font:600 11px var(--mono);letter-spacing:.06em;padding:3px 8px;border:1px solid currentColor;margin-bottom:18px}.ok{color:var(--good)}.fail{color:var(--bad)}
h3{font:600 11px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:26px 0 8px;border-bottom:1px solid var(--rule);padding-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 10px 7px 0;border-bottom:1px solid var(--rule);vertical-align:top}
th{font:600 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;padding-left:18px}
.dim{color:var(--muted)}.chip{font:600 11px var(--mono);padding:2px 6px;border:1px solid currentColor;text-transform:uppercase}
.bar{display:inline-block;height:8px;background:var(--cool);vertical-align:middle}.bar.neg{background:var(--accent)}
.call{border-left:3px solid var(--accent);background:var(--panel);padding:10px 14px;margin:8px 0}
svg text{font:11px var(--mono);fill:var(--muted)}.scroll{overflow-x:auto}
@media (max-width:820px){.wrap{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--rule);padding:12px 0 0}.slate{display:flex;overflow-x:auto}.slate button{flex:0 0 auto;width:auto;border-left:0;border-bottom:3px solid transparent}.slate button[aria-current=true]{border-bottom-color:var(--accent)}.figs{grid-template-columns:1fr 1fr}main{padding:20px 16px}}

.field{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 18px}
select,textarea,button.act{font:14px var(--serif);color:var(--ink);background:var(--panel);border:1px solid var(--rule);padding:6px 10px}
textarea{width:100%;min-height:110px;resize:vertical;line-height:1.5}
button.act{cursor:pointer;font:600 12px var(--mono);letter-spacing:.08em;text-transform:uppercase;background:var(--ink);color:var(--paper);border-color:var(--ink)}
label.f{font:600 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);display:flex;flex-direction:column;gap:4px}
mark{background:none;color:inherit;padding:0 3px;border-bottom:2px solid}
mark.ok{border-color:var(--good)}mark.unsupported{border-color:var(--bad);background:color-mix(in srgb,var(--bad) 14%,transparent)}
mark.misattributed{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 16%,transparent)}mark.style{border-color:var(--muted);border-bottom-style:dashed}
.legend{display:flex;gap:16px;flex-wrap:wrap;font:11px var(--mono);color:var(--muted);margin:8px 0 0}
.two{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:28px}@media (max-width:980px){.two{grid-template-columns:1fr}}
.pill{font:600 10px var(--mono);letter-spacing:.08em;text-transform:uppercase;padding:2px 6px;border:1px solid currentColor;margin-left:6px}
tr.sel td{background:var(--panel)}tr.click{cursor:pointer}tr.click:hover td{background:var(--panel)}
.wrap.single{grid-template-columns:1fr}
@media (max-width:820px){.wrap,.wrap.single{grid-template-columns:minmax(0,1fr)}aside,main{min-width:0}.slate{max-width:100vw}.figs{grid-template-columns:1fr 1fr}.fig{border-right:0;border-bottom:1px solid var(--rule)}}main{min-width:0}svg{max-width:100%;height:auto}@media (max-width:820px){th,td{padding:6px 6px 6px 0;font-size:13px}td.n,th.n{padding-left:6px}.bar2{max-width:44px}.chip{font-size:10px}}button.act.off{background:var(--panel);color:var(--ink);border-color:var(--rule)}.bar2{display:inline-block;height:10px;background:var(--cool);vertical-align:middle}.bar2.warn{background:var(--accent)}</style></head><body><div id="banner" class="banner" hidden>Synthetic demo data. The model here is a seeded simulator, not a real AI.</div><div class="wrap single"><main><div class="kicker">AI answer audit</div><h1 id="title">What do AI assistants say about your titles?</h1><div id="figs" class="figs"></div><section id="panel"></section></main></div><script>
"use strict";
const $=s=>document.querySelector(s);
function h(tag,attrs,...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs||{})){if(k==="class")e.className=v;else if(k.startsWith("on"))e.addEventListener(k.slice(2),v);else e.setAttribute(k,v)}
 for(const c of kids.flat(Infinity)){if(c==null||c===false)continue;e.append(c.nodeType?c:document.createTextNode(String(c)))}return e}
function sv(tag,attrs,...kids){const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const[k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);for(const c of kids.flat(Infinity))if(c!=null)e.append(typeof c==="string"?document.createTextNode(c):c);return e}
const api=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+" "+r.status);return r.json()};
const pct=(x,d=0)=>x==null?"n/a":(x*100).toFixed(d)+"%";const big=v=>v>=1e6?(v/1e6).toFixed(2)+"M":v>=1e3?(v/1e3).toFixed(0)+"K":Math.round(v)+"";


const state={model:"",models:[],report:null,sel:null,det:null,audits:[]};
const pc=(x,d=0)=>x==null?"n/a":(x*100).toFixed(d)+"%";
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
const VC={correct:"var(--good)",partial:"#b8860b",wrong:"var(--bad)",unknown:"var(--muted)"};
const PL={release:"When is it released?",cast:"Who stars in it?",about:"What is it about?",worth:"Is it worth watching?"};
function whisker(p,ci,w=110){const g=sv("svg",{width:w,height:14,viewBox:`0 0 ${w} 14`,role:"img","aria-label":`${pc(p)} (95% interval ${pc(ci[0])} to ${pc(ci[1])})`});
 g.append(sv("line",{x1:0,x2:w,y1:7,y2:7,stroke:"var(--rule)"}),sv("line",{x1:ci[0]*w,x2:ci[1]*w,y1:7,y2:7,stroke:"var(--cool)","stroke-width":3}),sv("circle",{cx:p*w,cy:7,r:4,fill:"var(--ink)"}));return g}
const cell=(p,ci)=>h("span",{style:"display:inline-flex;align-items:center;gap:8px;justify-content:flex-end"},pc(p),whisker(p,ci));
function figs(rep){const t=Object.values(rep.titles),rel=t.filter(r=>r.release),cast=t.filter(r=>r.cast),cat=t.filter(r=>r.category);
 const n=rel.reduce((s,r)=>s+r.release.n,0),ok=rel.reduce((s,r)=>s+r.release.correct,0);
 $("#figs").replaceChildren(...[["Release date correct",n?pc(ok/n):"n/a"],["Cast recall",cast.length?pc(mean(cast.map(r=>r.cast.recall))):"n/a"],["Possible fake names / answer",cast.length?mean(cast.map(r=>r.cast.possible_fabricated_names_per_answer)).toFixed(2):"n/a"],["Mention rate in genre lists",cat.length?pc(mean(cat.map(r=>r.category.mention_rate))):"n/a"]].map(([l,v])=>h("div",{class:"fig"},h("b",{},v),h("span",{},l))))}
function rows(rep){return Object.entries(rep.titles).map(([id,r])=>{const rel=r.release,cat=r.category;
 return h("tr",{class:"click"+(state.sel===id?" sel":""),onclick:()=>choose(id)},h("td",{},r.name,h("div",{class:"dim mono",style:"font-size:11px"},r.genre||"")),
  h("td",{class:"n"},rel?cell(rel.correct/rel.n,rel.correct_ci95):"n/a"),h("td",{class:"n"},r.cast?pc(r.cast.recall):"n/a"),h("td",{class:"n"},r.cast?r.cast.possible_fabricated_names_per_answer.toFixed(2):"n/a"),
  h("td",{class:"n"},cat?cell(cat.mention_rate,cat.mention_ci95):"n/a"),h("td",{class:"n"},cat&&cat.avg_position?cat.avg_position.toFixed(1):"n/a"),h("td",{class:"n"},cat?pc(cat.share_of_voice):"n/a"))})}
function pill(a,truth){if(a.verdict)return h("span",{class:"pill",style:`color:${VC[a.verdict]}`},a.verdict);
 if(a.cast_found)return [h("span",{class:"pill",style:`color:${a.cast_found.length===(truth.cast||[]).length?"var(--good)":"#b8860b"}`},`found ${a.cast_found.length}/${(truth.cast||[]).length}`),a.unverified_names.length?h("span",{class:"pill",style:"color:var(--bad)"},"unverified: "+a.unverified_names.join(", ")):null];
 if("mentioned" in a)return h("span",{class:"pill",style:`color:${a.mentioned?"var(--good)":"var(--bad)"}`},a.mentioned?"mentioned":"not mentioned");return null}
function inspector(d){if(!d)return h("div",{class:"call"},"Click a title to read every stored answer next to the ground truth.");const T=d.truth;
 const groups={};d.answers.forEach(a=>(groups[a.prompt_key]=groups[a.prompt_key]||[]).push(a));
 const order=["release","cast","category:"+d.genre,"worth","about"].filter(k=>groups[k]);
 return h("div",{},h("h3",{},"Answer inspector: "+d.title),
  h("div",{class:"call"},h("b",{},"Ground truth you entered: "),`released ${(T.release_date||["n/a"])[0]} · cast ${(T.cast||[]).join(", ")||"n/a"} · director ${(T.director||["n/a"])[0]}`),
  order.map(k=>{const L=groups[k];return h("div",{},h("h3",{},k.startsWith("category:")?`Genre list prompt: most anticipated ${d.genre} films`:PL[k]),
   L.slice(0,3).map(a=>h("div",{class:"call"},h("div",{class:"mono dim",style:"font-size:11px;margin-bottom:4px"},"sample "+a.sample,pill(a,T)),h("div",{style:"white-space:pre-wrap;font-size:14px"},a.answer))),
   L.length>3?h("div",{class:"dim mono",style:"font-size:12px"},`+${L.length-3} more stored samples`):null)}))}
function auditsView(audits){if(!audits||!audits.length)return null;
 return h("div",{style:"margin-top:28px"},h("h3",{},"Automated Audit History & Timeline"),
  h("div",{class:"dim mono",style:"font-size:12px;margin-bottom:8px"},`${audits.length} recorded audit snapshot(s)`),
  audits.slice(0,5).map(a=>{const m=a.metrics_summary||{};const rel=m.avg_release_correct!=null?pc(m.avg_release_correct):"n/a";const mr=m.avg_mention_rate!=null?pc(m.avg_mention_rate):"n/a";
   return h("div",{class:"call",style:"display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px"},
    h("div",{},h("b",{},`Audit #${a.id}: `),h("span",{class:"mono"},a.model),h("span",{class:"pill",style:"color:var(--good)"},a.status||"completed"),
     h("span",{class:"dim mono",style:"margin-left:8px;font-size:11px"},(a.started_at||"").slice(0,19).replace("T"," "))),
    h("div",{class:"mono",style:"font-size:12px"},`samples: ${a.samples_per_prompt} - calls: ${a.calls_made} - release ok: `,h("b",{},rel),` - genre mention: `,h("b",{},mr)));}))}
function render(){const rep=state.report;if(!Object.keys(rep.titles).length){$("#panel").replaceChildren(h("p",{class:"dim"},"No AI answers stored. Run `aivisibility run --yes`, or start with --demo."));return}
 figs(rep);
 const sel=state.models.length>1?h("label",{class:"f"},"Model",(()=>{const s=h("select",{onchange:async e=>{state.model=e.target.value;state.sel=null;state.det=null;await load()}},state.models.map(m=>h("option",{value:m},m)));s.value=state.model;return s})()):h("span",{class:"mono dim"},"model: "+state.model);
 $("#panel").replaceChildren(h("div",{class:"field"},sel,h("span",{class:"dim"},"Ranges are 95% Wilson intervals. Small samples are wide on purpose.")),
  h("div",{class:"scroll"},h("table",{},h("thead",{},h("tr",{},["Title","Release date correct","Cast recall","Fake names / answer","Mention rate in genre list","Avg rank","Share of voice"].map((x,i)=>h("th",{class:i?"n":""},x)))),h("tbody",{},rows(rep)))),inspector(state.det),auditsView(state.audits))}
async function load(){try{state.report=await api("/api/ai?model="+encodeURIComponent(state.model));try{state.audits=await api("/api/audits")}catch(x){state.audits=[]}render()}catch(e){$("#panel").replaceChildren(h("div",{class:"call fail"},"Could not load: "+e.message))}}
async function choose(id){state.sel=id;try{state.det=await api(`/api/answers/${id}?model=`+encodeURIComponent(state.model))}catch(e){state.det=null}render()}
(async()=>{const m=await api("/api/meta");if(m.simulated)$("#banner").hidden=false;state.models=m.models;state.model=m.models[0]||"";await load();
 const first=Object.keys((state.report||{titles:{}}).titles)[0];if(first)choose(first)})();

</script></body></html>"""
