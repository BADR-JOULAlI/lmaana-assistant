"""Génère PDF vectoriel et HTML autonome à partir du même contenu."""
from pathlib import Path
import sys
import re
import json
import html
import math

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tmp/pdfs/vendor'))
import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Preformatted, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon, Circle
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.pagesizes import A4

from course_data import PAGES
import course_middle
import course_engineering
import course_appendices

OUT = ROOT / 'output/course'
PDF = ROOT / 'output/pdf/Lmaana-Assistant-Cours-Illustre.pdf'
BLUE = colors.HexColor('#002FA7')
INK = colors.HexColor('#161616')
GRAY = colors.HexColor('#595959')
LINE = colors.HexColor('#D3D3D6')
BG = colors.HexColor('#F7F7F8')
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-Bold')

def label(d, x, y, text, size=20, bold=False, color=INK):
    for i, line in enumerate(text.split('\n')):
        d.add(String(x, y-i*(size+6), line, fontName='Arial-Bold' if bold else 'Arial', fontSize=size, fillColor=color))

def box(d, x, y, w, ht, title, body='', blue=False, dashed=False):
    d.add(Rect(x,y,w,ht,fillColor=BLUE if blue else BG,strokeColor=BLUE if blue else LINE,strokeWidth=1.2,strokeDashArray=[6,4] if dashed else None))
    label(d,x+17,y+ht-31,title,22,True,colors.white if blue else INK)
    if body: label(d,x+17,y+ht-64,body,17,False,colors.white if blue else GRAY)

def arrow(d,x1,y1,x2,y2):
    d.add(Line(x1,y1,x2,y2,strokeColor=BLUE,strokeWidth=2))
    ang=math.atan2(y2-y1,x2-x1); u=10
    d.add(Polygon([x2,y2,x2-u*math.cos(ang-.5),y2-u*math.sin(ang-.5),x2-u*math.cos(ang+.5),y2-u*math.sin(ang+.5)],fillColor=BLUE,strokeColor=BLUE))

def diagram(name):
    d=Drawing(1000,310)
    if name == 'cover':
        for i,(n,t,b) in enumerate([('01','Reconnaître','Audio -> texte\nLmaana 2.4 · prévu'),('02','Retrouver','Question -> preuves\nRecherche · active'),('03','Répondre','Preuves -> rédaction\nQwen · qualité à évaluer')]):
            x=i*342
            label(d,x+3,259,n,57,True,BLUE)
            box(d,x,50,316,160,t,b)
            if i<2: arrow(d,x+321,132,x+337,132)
    elif name == 'architecture':
        for x,t,b in [(0,'Streamlit','Question écrite\nPort 8501'),(343,'FastAPI','Pipeline + contrôles\nPort 8000'),(686,'Ollama + Qwen','Rédaction du candidat\nPort 11434')]:
            box(d,x,174,314,124,t,b)
        arrow(d,317,239,336,239); arrow(d,660,239,679,239)
        box(d,0,5,314,113,'Voix · prévue','Microphone -> Lmaana 2.4',dashed=True)
        box(d,343,5,314,113,'Qdrant local','Vecteurs + passages + sources')
        box(d,686,5,314,113,'Résultat','Contrôlé par FastAPI\nRéponse ou abstention')
        arrow(d,500,173,500,120); arrow(d,842,171,842,120)
        arrow(d,156,122,156,169)
    elif name == 'roles':
        box(d,0,176,314,117,'Modèle','Qwen3-4B\nPoids de génération',True)
        box(d,343,176,314,117,'Moteur','Ollama\nCharge et exécute les poids')
        box(d,686,176,314,117,'Application','Lmaana Assistant\nCherche, contrôle, affiche')
        arrow(d,319,234,337,234); arrow(d,662,234,680,234)
        box(d,0,9,486,116,'Autre tâche : la parole','Lmaana 2.4 : audio -> transcription')
        box(d,514,9,486,116,'Autre tâche : la recherche','Qdrant : vecteur -> passages associés')
        label(d,20,144,'Ne pas confondre identité du modèle et logiciel d’exécution.',18,color=BLUE)
    elif name == 'memory':
        label(d,0,280,'Sur disque',21,True); label(d,506,280,'Pendant l’inférence',21,True)
        box(d,0,60,430,186,'Fichier de poids','Q4_K_M : valeurs quantifiées\n+ échelles + métadonnées\nEnviron 2,5 Go dans ce projet')
        box(d,506,192,494,55,'Poids chargés',blue=True)
        box(d,506,126,494,55,'Cache KV : contexte et séquences')
        box(d,506,60,494,55,'Activations, buffers, affichage')
        arrow(d,440,151,492,151)
        label(d,0,17,'RAM système ~32 Go  |  VRAM GPU ~8 Go  |  Schéma non proportionnel',18,color=BLUE)
    elif name == 'ctc':
        label(d,0,281,'Temps audio ->',20,True)
        for y,title,tokens in [(183,'1. Choix par pas',['a','a','blanc','a','b','b']), (92,'2. Fusion des répétitions',['a','blanc','a','b']), (1,'3. Retrait des blancs',['a','a','b'])]:
            label(d,0,y+36,title,19,True)
            for i,t in enumerate(tokens):
                x=305+i*112
                d.add(Rect(x,y+4,100,60,fillColor=BLUE if t!='blanc' else BG,strokeColor=LINE))
                label(d,x+18,y+24,t,22,t!='blanc',colors.white if t!='blanc' else GRAY)
        label(d,789,29,'Résultat : aab',23,True,BLUE)
    elif name == 'language':
        box(d,0,179,305,118,'Question','Indices de script et mots\nLangue détectée')
        box(d,348,179,305,118,'Choix utilisateur','Auto ou langue explicite\nPriorité à la préférence')
        box(d,695,179,305,118,'Langue cible','fr / ar / ary\nary-Latn / en')
        arrow(d,313,239,340,239); arrow(d,661,239,686,239)
        box(d,0,13,305,115,'Indéterminée','Demander une précision',dashed=True)
        box(d,348,13,305,115,'Sources','Peuvent être dans\nune autre langue')
        box(d,695,13,305,115,'Génération','Rédiger puis contrôler\nFormat + fidélité + langue',True)
        arrow(d,151,170,151,134); arrow(d,847,170,847,134); arrow(d,661,70,686,70)
    elif name == 'source':
        items=[('Collecter','URL + date\n+ contenu'),('Identifier','Hash + pages\n+ éditeur'),('Vérifier','Revue + portée\n+ échéance'),('Autoriser','Valide aujourd’hui\net contenu identique')]
        for i,(t,b) in enumerate(items):
            x=i*252; box(d,x,140,244,145,t,b,blue=i==3)
            if i<3: arrow(d,x+245,213,x+251,213)
        box(d,0,4,487,95,'Historique ou non vérifiée','Référence séparée, pas preuve affirmative')
        box(d,514,4,486,95,'Expirée ou contenu modifié','Nouvelle revue avant réutilisation')
        arrow(d,614,132,740,108)
    elif name == 'chunks':
        label(d,0,284,'Document fictif : « Photo nécessaire, sauf pour les comptes déjà vérifiés. »',20,True)
        box(d,0,127,480,115,'Coupe mécanique','Chunk A : Photo nécessaire\nChunk B : sauf pour les comptes...',dashed=True)
        box(d,520,127,480,115,'Frontière de sens','Un chunk : règle + exception',True)
        label(d,0,77,'Le chunk A peut devenir une obligation sans exception.',20,color=GRAY)
        label(d,0,37,'Overlap : utile aux frontières, mais ne remplace pas une section complète.',20,color=BLUE)
    elif name == 'vectors':
        box(d,0,168,310,130,'Question q','(1, 1, 0)\nLongueur : racine(2)',True)
        box(d,344,168,310,130,'Passage d1','(1, 1, 0)\nCosinus avec q : 1')
        box(d,687,168,313,130,'Passage d2','(1, 0, 1)\nCosinus avec q : 0,5')
        arrow(d,314,237,339,237); arrow(d,660,237,682,237)
        label(d,0,121,'Plus proche dans cet espace',20,True)
        d.add(Rect(0,58,480,32,fillColor=BLUE,strokeColor=None)); label(d,493,66,'d1 : 1',21,True)
        d.add(Rect(0,8,240,32,fillColor=BLUE,strokeColor=None)); label(d,252,16,'d2 : 0,5',21,True)
        label(d,658,77,'Exemple mathématique,',18,color=GRAY); label(d,658,48,'pas score mesuré du projet.',18,color=GRAY)
    elif name == 'validation':
        for x,t,b in [(0,'Preuves admises','Sources + pertinence'),(344,'Qwen','Sortie JSON'),(688,'Contrôles','Structure, citations, langue')]: box(d,x,184,312,111,t,b)
        arrow(d,317,236,337,236); arrow(d,661,236,681,236)
        box(d,0,7,312,117,'Aucune preuve','Abstention expliquée')
        box(d,344,7,312,117,'Invalide','Une nouvelle tentative\nSinon : réponse non validée',dashed=True)
        box(d,688,7,312,117,'Valide aux contrôles','Réponse + références\nVérification humaine possible',True)
        arrow(d,150,177,150,130); arrow(d,845,177,845,130); arrow(d,755,175,572,134)
    elif name == 'concurrency':
        box(d,0,182,312,115,'Requête A','Créneau libre -> calcul',True)
        box(d,344,182,312,115,'Requête B','Créneau occupé -> 429')
        box(d,688,182,312,115,'Timeout de A','504 : attente dépassée')
        arrow(d,319,238,337,238); arrow(d,663,238,681,238)
        box(d,0,12,485,122,'Cas normal','Fin du calcul -> libération du créneau')
        box(d,516,12,484,122,'Après timeout','État indisponible -> 503\nDiagnostic avant redémarrage')
        arrow(d,151,173,151,141); arrow(d,844,173,844,141)
    elif name == 'tests':
        bands=[(0,6,1000,'Unitaires : une règle, un cas limite'),(70,82,860,'Intégration / contrats : liens entre composants'),(140,158,720,'Bout en bout : scénario utilisateur'),(210,234,580,'Qualité réelle : langues et faits annotés')]
        for x,y,w,t in bands:
            d.add(Rect(x,y,w,62,fillColor=BLUE if y==234 else BG,strokeColor=LINE))
            label(d,x+18,y+23,t,21,True,colors.white if y==234 else INK)
    return d

FIGS={b[1]:diagram(b[1]) for page in PAGES for b in page['blocks'] if b[0]=='fig'}
FIG_IDS={name:i+1 for i,name in enumerate(FIGS)}
def svg_text(name,d):
    svg=renderSVG.drawToString(d)
    svg=svg.replace('font-family: Arial-Bold;', 'font-family: Arial; font-weight: 700;')
    svg=svg.replace('<title>...</title>',f'<title>Infographie {FIG_IDS[name]:02d} - {name}</title>')
    svg=svg.replace('<desc>...</desc>','<desc>Schéma pédagogique du cours Lmaana Assistant.</desc>')
    # Les identifiants restent uniques quand tous les SVG sont inclus dans le HTML.
    svg=svg.replace('id="clip"',f'id="clip-{name}"').replace('url(#clip)',f'url(#clip-{name})')
    svg=svg.replace('id="group"',f'id="group-{name}"')
    return svg

FIG_SVGS={name:svg_text(name,d) for name,d in FIGS.items()}

def para(text, size=10.4, leading=None, color=INK, **kwargs):
    text=re.sub(r' ([;:!?])', '\u00a0'+r'\1', text)
    return Paragraph(text, ParagraphStyle('local',fontName='Arial',fontSize=size,leading=leading or size*1.38,textColor=color,spaceAfter=8,allowWidows=0,allowOrphans=0,**kwargs))

def block_flow(block, size, width):
    kind=block[0]; out=[]
    if kind=='p': out=[para(block[1],size)]
    elif kind=='h': out=[Spacer(1,4),para('<b>'+block[1]+'</b>',size+1.3)]
    elif kind=='refs': out=[Spacer(1,1),para(block[1],size-1.3,color=GRAY)]
    elif kind=='note':
        t=Table([[para(block[1],size-.15)]],colWidths=[width])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),BG),('LINEBEFORE',(0,0),(0,-1),2,BLUE),('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
        out=[t,Spacer(1,9)]
    elif kind=='code':
        sty=ParagraphStyle('code',fontName='Arial',fontSize=size-.9,leading=(size-.9)*1.4,textColor=BLUE)
        pf=Preformatted(block[1],sty)
        t=Table([[pf]],colWidths=[width]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),BG),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
        out=[t,Spacer(1,10)]
    elif kind=='table':
        headers,rows=block[1:];n=len(headers)
        ratios={2:[.29,.71],3:[.25,.35,.40],5:[.25,.20,.23,.17,.15]}.get(n,[1/n]*n)
        data=[[para('<b>'+html.escape(str(x))+'</b>',size-.65,color=colors.white) for x in headers]]
        data += [[para(html.escape(str(x)),size-.8) for x in row] for row in rows]
        t=Table(data,colWidths=[r*width for r in ratios],hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,1),(-1,-1),.35,LINE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,BG])]))
        out=[t,Spacer(1,10)]
    elif kind=='fig':
        d=FIGS[block[1]];scale=width/1000
        fresh=Drawing(width,d.height*scale);fresh.add(d);fresh.scale(scale,scale)
        out=[fresh,Spacer(1,4),para(f'<b>Figure {FIG_IDS[block[1]]:02d}.</b> '+block[2],size-1.5,color=GRAY),Spacer(1,5)]
    elif kind=='ar':
        shaped=get_display(arabic_reshaper.reshape(block[1]))
        out=[para(shaped,22,30,alignment=2),para(block[2],size-1,color=GRAY)]
    return out

def build_pdf():
    PDF.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(PDF),pagesize=A4,pageCompression=1)
    c.setTitle('Comprendre et construire Lmaana Assistant - Cours illustré')
    c.setAuthor('Cours préparé pour le projet Lmaana Assistant')
    c.setSubject('ASR, Darija, RAG, LLM locaux, recherche, architecture et évaluation')
    W,H=A4; margin=43; width=W-2*margin;qa=[]
    for i,item in enumerate(PAGES,1):
        c.bookmarkPage(f'p{i}');c.addOutlineEntry(f'{item["part"]} - {item["title"].replace(chr(10)," ")}',f'p{i}',0,False)
        c.setFillColor(GRAY);c.setFont('Arial',8)
        c.drawString(margin,H-25,'LMAANA ASSISTANT  /  COURS ILLUSTRÉ')
        c.drawRightString(W-margin,H-25,'03 OCTOBRE 2026')
        c.setStrokeColor(LINE);c.setLineWidth(.5);c.line(margin,H-33,W-margin,H-33)
        y=H-57
        c.setFillColor(BLUE);c.setFont('Arial-Bold',9);c.drawString(margin,y,item['part']);y-=15
        title=para('<b>'+item['title'].replace('\n','<br/>')+'</b>',31 if i==1 else 25,34 if i==1 else 28)
        tw,th=title.wrap(width,1000);title.drawOn(c,margin,y-th);y-=th+10
        sub=para(item['subtitle'],11,15,color=GRAY);sw,sh=sub.wrap(width,1000);sub.drawOn(c,margin,y-sh);y-=sh+18
        chosen=None
        for size in [10.4,10.2,10,9.8,9.6]:
            flows=[f for b in item['blocks'] for f in block_flow(b,size,width)]
            measures=[(f,*f.wrap(width,2000)) for f in flows]
            total=sum(ht+f.getSpaceAfter() for f,ww,ht in measures)
            if total<=y-49:
                chosen=(size,measures,total);break
        if not chosen: raise ValueError(f'Page {i} overflow: {total:.1f} available {y-49:.1f}: {item["title"]}')
        size,measures,total=chosen
        for f,ww,ht in measures:
            y-=ht;f.drawOn(c,margin,y);y-=f.getSpaceAfter()
        c.setStrokeColor(LINE);c.line(margin,34,W-margin,34)
        c.setFont('Arial',8);c.setFillColor(GRAY);c.drawString(margin,21,'Cours technique · Les exemples ne constituent pas un conseil administratif.')
        c.setFillColor(BLUE);c.setFont('Arial-Bold',10);c.drawRightString(W-margin,20,f'{i:02d} / {len(PAGES):02d}')
        qa.append({'page':i,'title':item['title'],'body_font':size,'bottom':round(y,2)})
        c.showPage()
    c.save()
    return qa

def block_html(b):
    k=b[0]
    if k=='p': return '<p>'+b[1]+'</p>'
    if k=='h': return '<h3>'+b[1]+'</h3>'
    if k=='note': return '<aside class="note">'+b[1]+'</aside>'
    if k=='refs': return '<p class="refs">'+b[1]+'</p>'
    if k=='code': return '<pre><code>'+html.escape(b[1])+'</code></pre>'
    if k=='ar': return '<p lang="ary" dir="rtl" class="arabic">'+b[1]+'</p><p class="refs">'+b[2]+'</p>'
    if k=='fig': return '<figure role="img" aria-label="'+html.escape(b[2],quote=True)+'">'+re.sub(r'<\?xml.*?\?>|<!DOCTYPE.*?>','',FIG_SVGS[b[1]],flags=re.S)+'<figcaption>Figure '+str(FIG_IDS[b[1]]).zfill(2)+'. '+b[2]+'</figcaption></figure>'
    if k=='table':
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in b[1])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in b[2])+'</tbody></table></div>'
    raise ValueError(k)

def build_html():
    nav=''.join(f'<a href="#p{i}"><span>{i:02d}</span> {html.escape(v["title"].replace(chr(10)," "))}</a>' for i,v in enumerate(PAGES,1))
    sections=''.join(f'<section id="p{i}" aria-labelledby="t{i}"><div class="part">{v["part"]}</div><h2 id="t{i}">{v["title"].replace(chr(10),"<br/>")}</h2><p class="subtitle">{v["subtitle"]}</p>'+''.join(block_html(b) for b in v['blocks'])+f'<footer>Page du PDF : {i:02d} / {len(PAGES):02d} <a href="#sommaire">Sommaire</a></footer></section>' for i,v in enumerate(PAGES,1))
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#F7F7F8;color:#161616;font-family:Arial,"Helvetica Neue",sans-serif;font-size:17px;line-height:1.6}a{color:#002FA7;text-underline-offset:3px}header{background:#FFFFFF;border-bottom:1px solid #D3D3D6;padding:24px 36px}header h1{font-size:20px;margin:0}header p{font-size:14px;margin:7px 0 0;color:#595959}.layout{display:grid;grid-template-columns:270px minmax(0,900px);gap:36px;max-width:1280px;margin:auto}nav{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:25px 16px 35px;font-size:13px}nav h2{font-size:16px}nav a{display:block;text-decoration:none;border-top:1px solid #D3D3D6;padding:9px 0;color:#161616}nav a:hover,nav a:focus{color:#002FA7;text-decoration:underline}nav span{color:#002FA7;font-weight:bold;margin-right:7px}main{min-width:0}section{background:#FFFFFF;margin:30px 0;padding:48px;border:1px solid #D3D3D6;scroll-margin-top:20px}.part{color:#002FA7;font-size:13px;letter-spacing:.06em;font-weight:bold}h2{font-size:36px;line-height:1.13;letter-spacing:-.03em;margin:13px 0}#p1 h2{font-size:49px}.subtitle{font-size:19px;color:#595959;margin-top:15px;margin-bottom:30px}h3{font-size:21px;line-height:1.25;margin:26px 0 12px}p{margin:0 0 18px}.note{background:#F7F7F8;border-left:3px solid #002FA7;padding:17px 20px;font-size:16px;margin:22px 0}.refs,figcaption{font-size:13px;line-height:1.5;color:#595959}figure{margin:25px 0}figure svg{display:block;width:100%;height:auto}figcaption{margin-top:10px}pre{background:#F7F7F8;padding:16px;border-top:1px solid #D3D3D6;color:#002FA7;font-size:14px;overflow:auto;line-height:1.6}code{font-family:Arial,sans-serif}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;margin:20px 0}th{text-align:left;background:#002FA7;color:#FFFFFF}td,th{padding:10px 12px;border-bottom:1px solid #D3D3D6;vertical-align:top}tr:nth-child(even){background:#F7F7F8}.table-wrap{overflow-x:auto}.arabic{font-size:32px;line-height:1.7}footer{margin-top:35px;padding-top:13px;border-top:1px solid #D3D3D6;color:#595959;font-size:12px;display:flex;justify-content:space-between}button{font-family:inherit;background:#002FA7;border:0;color:#FFFFFF;padding:10px 17px;font-size:14px;cursor:pointer;margin-top:12px}a:focus-visible,button:focus-visible{outline:3px solid #002FA7;outline-offset:4px}@media(max-width:850px){.layout{display:block}nav{position:relative;max-height:260px;border-bottom:1px solid #D3D3D6;margin:0 18px}section{margin:18px;padding:27px}h2{font-size:29px}#p1 h2{font-size:36px}header{padding:22px}body{font-size:16px}figure{overflow-x:auto}figure svg{min-width:550px}footer{gap:12px}}@media print{body{background:#FFFFFF;font-size:11pt}header,nav,button,footer{display:none}.layout{display:block}section{break-before:page;border:0;margin:0;padding:0}#p1{break-before:auto}h2{font-size:23pt}#p1 h2{font-size:30pt}figure svg{min-width:0}a{color:inherit;text-decoration:none}}'''
    content='<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Lmaana Assistant - Cours illustré</title><style>'+css+'</style></head><body><header><h1>Lmaana Assistant · Cours illustré</h1><p>31 chapitres · 12 infographies · 14 exercices corrigés · État du projet au 3 octobre 2026</p><button onclick="window.print()">Imprimer cette version</button></header><div class="layout"><nav id="sommaire" aria-label="Sommaire"><h2>Sommaire</h2>'+nav+'</nav><main>'+sections+'</main></div></body></html>'
    (OUT/'Lmaana-Assistant-Cours.html').write_text(content,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qa=build_pdf();build_html()
    (OUT/'infographies').mkdir(exist_ok=True)
    for name,svg in FIG_SVGS.items(): (OUT/'infographies'/f'{FIG_IDS[name]:02d}-{name}.svg').write_text(svg,encoding='utf-8')
    words=len(re.findall(r"\b[\w’'-]+\b",re.sub('<[^>]+>',' ',json.dumps(PAGES,ensure_ascii=False))))
    (ROOT/'tmp/pdfs/course-layout.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pdf':str(PDF),'pages':len(PAGES),'figures':len(FIGS),'approx_content_words':words,'minimum_body_font':min(x['body_font'] for x in qa),'lowest_content_y':min(x['bottom'] for x in qa)},ensure_ascii=False))

if __name__=='__main__': main()
