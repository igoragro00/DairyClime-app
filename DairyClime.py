#Aplicativo para Avaliação do Conforto Térmico de Bovinos Leiteiros

# ============================================================
# DairyClime – Conforto Térmico (ITU) para Vacas de Leite
# App Streamlit + NASA POWER + Relatório PDF com gráfico
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from datetime import datetime, timedelta, date
from io import BytesIO

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="DairyClime", page_icon="🐄", layout="wide")

# ============================================================
# FUNÇÕES
# ============================================================
def calcular_itu(ta_c, ur_pct):
    """
    ITU = 0,8×Ta + (UR×(Ta−14,3))/100 + 46,3
    Ta em °C; UR em %
    """
    return 0.8 * ta_c + (ur_pct * (ta_c - 14.3)) / 100.0 + 46.3


def classificar_itu(itu):
    # Escala alinhada com o que você vinha usando
    if itu <= 70:
        return "Normal"
    elif itu <= 78:
        return "Alerta"
    elif itu <= 82:
        return "Perigo"
    else:
        return "Emergência"


def recomendacao_por_classe(classe):
    # Linguagem acessível (produtor/técnico)
    if classe == "Normal":
        return "✅ Conforto: manter água limpa e sombra. Rotina normal."
    if classe == "Alerta":
        return "🟡 Atenção: reforçar água e sombra. Evitar manejo no horário mais quente."
    if classe == "Perigo":
        return "🟠 Risco alto: usar ventilação e/ou aspersão se disponível. Mudar manejo para manhã cedo/tarde."
    return "🔴 Emergência: ação imediata! Levar animais para sombra, muita água e resfriamento (ventilador/aspersor). Evitar qualquer manejo."


def cor_por_itu(itu):
    if itu <= 70:
        return "#2E7D32"  # verde
    elif itu <= 78:
        return "#F9A825"  # amarelo
    elif itu <= 82:
        return "#EF6C00"  # laranja
    else:
        return "#C62828"  # vermelho


def diagnostico_periodo(media_itu, p_alerta, p_perigo, p_emerg):
    # Texto automático resumido
    if media_itu <= 70:
        return (f"No geral, o período ficou em conforto térmico (ITU médio {media_itu:.1f}). "
                "Mesmo assim, continue garantindo água e sombra.")
    if media_itu <= 78:
        return (f"O período teve alerta (ITU médio {media_itu:.1f}). "
                f"Tivemos {p_alerta:.1f}% dos dias em alerta. Reforce sombra e água.")
    if media_itu <= 82:
        return (f"O período teve risco alto de estresse térmico (ITU médio {media_itu:.1f}). "
                f"Tivemos {p_perigo:.1f}% dos dias em perigo. Intensifique resfriamento e evite manejo no calor.")
    return (f"O período foi crítico (ITU médio {media_itu:.1f}). "
            f"Tivemos {p_emerg:.1f}% dos dias em emergência. Priorize resfriamento imediato e constante.")


@st.cache_data(show_spinner=False)
def obter_dados_nasa_power(lat, lon, data_ini_yyyymmdd, data_fim_yyyymmdd):
    """
    Busca Ta (T2M) e UR (RH2M) diários na NASA POWER.
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M"
        f"&community=AG"
        f"&latitude={lat}&longitude={lon}"
        f"&start={data_ini_yyyymmdd}&end={data_fim_yyyymmdd}"
        f"&format=JSON"
    )

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()

    params = js["properties"]["parameter"]
    t2m = params.get("T2M", {})
    rh2m = params.get("RH2M", {})

    df = pd.DataFrame({
        "Data": list(t2m.keys()),
        "Ta": list(t2m.values()),
        "UR": list(rh2m.values())
    })

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Ta"] = pd.to_numeric(df["Ta"], errors="coerce")
    df["UR"] = pd.to_numeric(df["UR"], errors="coerce")

    df = df.dropna(subset=["Data", "Ta", "UR"]).sort_values("Data").reset_index(drop=True)
    return df


def preparar_df_plot(df, data_ini, data_fim):
    """
    Define o tipo de gráfico conforme tamanho do período:
    - <= 15 dias: diário
    - 16..90: média 5 dias
    - 91..364: mensal por mês/ano
    - >= 365: climatologia mensal (Jan..Dez, média de todos os anos)
    """
    dias = (data_fim - data_ini).days + 1

    if dias <= 15:
        df_plot = df[["Data", "ITU"]].copy()
        df_plot["Label"] = df_plot["Data"].dt.strftime("%d/%m")
        titulo = "ITU Diário"

    elif dias <= 90:
        df_plot = (
            df.set_index("Data")[["ITU"]]
              .resample("5D")
              .mean()
              .reset_index()
        )
        df_plot["Label"] = df_plot["Data"].dt.strftime("%d/%m")
        titulo = "ITU (Média a cada 5 dias)"

    elif dias < 365:
        df_plot = (
            df.set_index("Data")[["ITU"]]
              .resample("M")
              .mean()
              .reset_index()
        )
        df_plot["Label"] = df_plot["Data"].dt.strftime("%b/%Y")  # ex: Jan/2022
        titulo = "ITU Médio Mensal"

    else:
        # climatologia mensal (12 barras) — ideal para muitos anos
        df_tmp = df.copy()
        df_tmp["Mes"] = df_tmp["Data"].dt.month
        df_plot = (
            df_tmp.groupby("Mes", as_index=False)["ITU"]
                  .mean()
        )
        mapa_mes = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
        df_plot["Label"] = df_plot["Mes"].map(mapa_mes)
        titulo = "ITU Médio Mensal (média de todos os anos)"

    return df_plot, titulo


def plot_barras_itu(df_plot, titulo, altura=4.2, largura=9.2):
    """
    Retorna um matplotlib Figure com barras coloridas por classe
    e valores centralizados em branco.
    """
    fig, ax = plt.subplots(figsize=(largura, altura))

    cores = [cor_por_itu(v) for v in df_plot["ITU"]]
    bars = ax.bar(df_plot["Label"], df_plot["ITU"], color=cores)

    # Linhas de referência
    ax.axhline(70, linestyle="--", linewidth=1)
    ax.axhline(78, linestyle="--", linewidth=1)
    ax.axhline(82, linestyle="--", linewidth=1)

    ax.set_title(titulo)
    ax.set_ylabel("ITU")

    # Ajustes eixo X: não poluir
    max_labels = 18
    if len(df_plot) > max_labels:
        step = int(np.ceil(len(df_plot) / max_labels))
        for i, label in enumerate(ax.get_xticklabels()):
            if i % step != 0:
                label.set_visible(False)

    # Valores no meio da barra (branco + negrito)
    for bar, val in zip(bars, df_plot["ITU"]):
        h = bar.get_height()
        # só escreve se a barra for “alta o suficiente” para não virar bagunça
        if h >= 10:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h/2,
                f"{val:.1f}",
                ha="center", va="center",
                color="white",
                fontsize=9,
                fontweight="bold"
            )

    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def gerar_pdf_relatorio(nome_local, lat, lon, data_ini, data_fim,
                        media_itu, classe_media, diag_texto,
                        p_alerta, p_perigo, p_emerg,
                        fig_matplotlib):
    """
    Gera PDF e embute o gráfico (mesmas cores).
    """
    if nome_local is None or str(nome_local).strip() == "":
        nome_local = "Local não informado"

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    y = h - 50

   # =========================
# Cabeçalho
# =========================
c.setFont("Helvetica-Bold", 16)
c.drawString(40, y, "DairyClime – Relatório de Conforto Térmico (Vacas de Leite)")
y -= 30

# Texto institucional do app
c.setFont("Helvetica", 10)
texto_descricao = (
    "DairyClime é um aplicativo desenvolvido no âmbito de projetos de ensino, pesquisa "
    "e extensão universitária, vinculado à Universidade Federal do Maranhão (UFMA) e à UNESP, "
    "com apoio de laboratórios e pesquisadores das áreas de Zootecnia e Ciências Agrárias.\n\n"
    "A ferramenta utiliza dados climáticos da NASA/POWER para avaliar o conforto térmico "
    "de vacas de leite, auxiliando produtores e técnicos na tomada de decisão para o manejo "
    "do estresse térmico."
)

text_obj = c.beginText(40, y)
for linha in texto_descricao.split("\n"):
    text_obj.textLine(linha)

c.drawText(text_obj)

# Ajuste do cursor vertical após o bloco
y -= 70


    # Destaque do resultado (tipo “card” simples)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, f"ITU médio do período: {media_itu:.1f}")
    y -= 18
    c.drawString(40, y, f"Classificação: {classe_media}")
    y -= 22

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Diagnóstico do período:")
    y -= 16
    c.setFont("Helvetica", 11)

    # Quebra de linha manual simples
    max_chars = 95
    linhas = [diag_texto[i:i+max_chars] for i in range(0, len(diag_texto), max_chars)]
    for ln in linhas[:4]:
        c.drawString(50, y, ln)
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Frequência no período:")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Alerta: {p_alerta:.1f}% dos dias")
    y -= 14
    c.drawString(50, y, f"Perigo: {p_perigo:.1f}% dos dias")
    y -= 14
    c.drawString(50, y, f"Emergência: {p_emerg:.1f}% dos dias")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Recomendação principal:")
    y -= 16
    c.setFont("Helvetica", 11)
    rec = recomendacao_por_classe(classe_media)
    linhas = [rec[i:i+max_chars] for i in range(0, len(rec), max_chars)]
    for ln in linhas[:3]:
        c.drawString(50, y, ln)
        y -= 14

    # Inserir gráfico do app no PDF (mesmas cores/layout)
    # 1) salvar fig em PNG em memória
    img_buf = BytesIO()
    fig_matplotlib.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
    img_buf.seek(0)

    # 2) colocar nova página e desenhar imagem
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 50, "Gráfico do ITU (mesmas cores do aplicativo)")

    img = ImageReader(img_buf)

    # dimensões do gráfico no PDF
    img_w = w - 80
    img_h = 320
    c.drawImage(img, 40, h - 50 - 20 - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')

    # Rodapé
    c.setFont("Helvetica", 9)
    c.drawString(40, 60, "Fonte dos dados: NASA POWER (dados diários; atraso médio ~4 dias; resolução ~55 km).")
    c.drawString(40, 48, "Fórmula do ITU: ITU = 0,8×Ta + (UR×(Ta−14,3))/100 + 46,3  |  Ta(°C)  UR(%).")
    c.drawString(40, 36, "Responsáveis: Prof. Dr. Kamila Cunha de Meneses (UFMA) | Msc. Igor C. O. Vieira (FCAV/UNESP).")

    c.save()
    buffer.seek(0)
    return buffer


# ============================================================
# SIDEBAR (melhorado e acessível)
# ============================================================
st.sidebar.title("🐄 DairyClime")

st.sidebar.markdown(
    """
**O que é conforto térmico?**  
É quando a vaca consegue manter a temperatura do corpo sem esforço.  
Quando está **quente e úmido**, ela tem dificuldade de perder calor → pode **comer menos** e **produzir menos leite**.

**O que é ITU (Índice de Temperatura e Umidade)?**  
É um número que junta **Temperatura (°C)** e **Umidade (%)** para indicar o risco de estresse térmico.

**Fórmula usada:**  
ITU = 0,8×Ta + (UR×(Ta−14,3))/100 + 46,3  
Ta = temperatura do ar (°C) | UR = umidade relativa (%)

**Classes (vacas de leite):**
- 🟢 Normal: ITU ≤ 70  
- 🟡 Alerta: 70 < ITU ≤ 78  
- 🟠 Perigo: 78 < ITU ≤ 82  
- 🔴 Emergência: ITU > 82  

**Fonte dos dados:** NASA POWER  
- Dados diários globais
- Resolução ~55 km
- Atraso médio ~4 dias
"""
)

st.sidebar.markdown(
    """
**Responsáveis:**  
- Prof. Dr. Kamila Cunha de Meneses – UFMA  
- Msc. Igor Cristian de Oliveira Vieira – FCAV/UNESP  

LinkedIn:  
- Igor: https://www.linkedin.com/in/eng-igor-vieira/  
- Kamila: https://www.linkedin.com/in/kamila-cunha-de-meneses-38008586/
"""
)

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
st.title("🌡️ DairyClime – Conforto Térmico (ITU) para Vacas de Leite")

modo = st.radio("Selecione o modo:", ["Automático (NASA POWER)", "Manual (Ta e UR)"], horizontal=True)

# ============================================================
# MODO MANUAL
# ============================================================
if modo == "Manual (Ta e UR)":
    st.subheader("🧮 Cálculo manual do ITU (para um dia específico)")

    c1, c2 = st.columns(2)
    ta = c1.number_input("Temperatura do ar (°C)", value=30.0, step=0.5)
    ur = c2.number_input("Umidade relativa (%)", value=60.0, step=1.0, min_value=0.0, max_value=100.0)

    itu = calcular_itu(ta, ur)
    classe = classificar_itu(itu)

    st.markdown(
        f"""
        <div style="padding:14px;border-radius:10px;background:#f7f7f7;border-left:10px solid {cor_por_itu(itu)}">
            <div style="font-size:14px;font-weight:700;">ITU calculado</div>
            <div style="font-size:44px;font-weight:800;line-height:1">{itu:.1f}</div>
            <div style="font-size:18px;font-weight:800;margin-top:4px">{classe}</div>
            <div style="margin-top:6px;font-size:14px"><b>O que fazer:</b> {recomendacao_por_classe(classe)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# MODO AUTOMÁTICO
# ============================================================
st.subheader("📍 Localização (Automático)")
c1, c2 = st.columns(2)
lat = c1.text_input("Latitude", "-5.0")
lon = c2.text_input("Longitude", "-45.0")

nome_local = st.text_input("Nome do local (opcional – aparece no relatório)", value="")

st.subheader("📅 Período de análise")

hoje = datetime.today().date()
limite_max = hoje - timedelta(days=4)

data_ini = st.date_input(
    "Data inicial",
    min_value=date(1990, 1, 1),
    max_value=limite_max,
    value=date(2020, 1, 1)
)

data_fim = st.date_input(
    "Data final",
    min_value=data_ini,
    max_value=limite_max,
    value=limite_max
)

# Bloqueio do atraso de 4 dias (NASA POWER)
if data_fim > limite_max:
    st.error("⚠️ O NASA POWER tem atraso médio de ~4 dias. Selecione uma data final até "
             f"{limite_max.strftime('%d/%m/%Y')}.")
    st.stop()

if st.button("🔍 Analisar Conforto Térmico"):

    # valida coordenadas
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except ValueError:
        st.error("Digite coordenadas válidas (ex.: -5.1234).")
        st.stop()

    with st.spinner("Buscando dados diários do NASA POWER..."):
        try:
            df = obter_dados_nasa_power(
                lat_f, lon_f,
                data_ini.strftime("%Y%m%d"),
                data_fim.strftime("%Y%m%d")
            )
        except Exception as e:
            st.error("Erro ao buscar dados no NASA POWER. Verifique as coordenadas e tente novamente.")
            st.exception(e)
            st.stop()

    if df.empty:
        st.warning("Não foi possível obter dados para este local/período.")
        st.stop()

    # Cálculo ITU
    df["ITU"] = calcular_itu(df["Ta"], df["UR"])
    df["Classe"] = df["ITU"].apply(classificar_itu)

    media_itu = float(df["ITU"].mean())
    classe_media = classificar_itu(media_itu)

    # Percentuais
    p_alerta = (df["Classe"] == "Alerta").mean() * 100
    p_perigo = (df["Classe"] == "Perigo").mean() * 100
    p_emerg = (df["Classe"] == "Emergência").mean() * 100

    diag = diagnostico_periodo(media_itu, p_alerta, p_perigo, p_emerg)

    # Preparar plot agregado (corrigido)
    df_plot, titulo = preparar_df_plot(df, data_ini, data_fim)

    # Gráfico (tamanho consistente e legível)
    fig = plot_barras_itu(df_plot, titulo, altura=4.2, largura=9.2)

    # Mostrar período sem poluir eixos
    st.caption(f"Período selecionado: {data_ini.strftime('%d/%m/%Y')} → {data_fim.strftime('%d/%m/%Y')}")

    # Exibir gráfico
    st.pyplot(fig, clear_figure=False)

    # CARD do resultado
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:10px;background:#f7f7f7;border-left:10px solid {cor_por_itu(media_itu)}">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:14px;font-weight:700;">ITU médio do período</div>
                    <div style="font-size:46px;font-weight:900;line-height:1">{media_itu:.1f}</div>
                    <div style="font-size:18px;font-weight:900;margin-top:4px">{classe_media}</div>
                </div>
                <div style="max-width:520px;font-size:14px">
                    <div><b>Diagnóstico:</b> {diag}</div>
                    <div style="margin-top:6px"><b>O que fazer:</b> {recomendacao_por_classe(classe_media)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Frequência de estresse térmico no período")
    cA, cP, cE = st.columns(3)
    cA.metric("🟡 Alerta", f"{p_alerta:.1f}%")
    cP.metric("🟠 Perigo", f"{p_perigo:.1f}%")
    cE.metric("🔴 Emergência", f"{p_emerg:.1f}%")

    st.markdown("### 📄 Relatório em PDF (com o mesmo gráfico do app)")

    pdf_buf = gerar_pdf_relatorio(
        nome_local=nome_local,
        lat=lat_f, lon=lon_f,
        data_ini=data_ini, data_fim=data_fim,
        media_itu=media_itu,
        classe_media=classe_media,
        diag_texto=diag,
        p_alerta=p_alerta, p_perigo=p_perigo, p_emerg=p_emerg,
        fig_matplotlib=fig
    )

    st.download_button(
        label="📥 Baixar relatório em PDF",
        data=pdf_buf,
        file_name="DairyClime_Relatorio.pdf",
        mime="application/pdf"
    )

