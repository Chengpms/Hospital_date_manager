# -*- coding: utf-8 -*-
"""
mc_4_animado: Motor de Eventos Discretos con Llegadas Estocásticas (Johnson SU),
Descanso Flexible y Citas Dinámicas.
"""

import os
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from scipy.stats import johnsonsu

print("🏥 INICIANDO SIMULACIÓN DE EVENTOS DISCRETOS (DISTRIBUCIÓN JOHNSON SU)...")
print("-" * 75)

# --- CONFIGURACIÓN DE RUTAS ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELO_JOBLIB_PATH = MODELS_DIR / "modelo_definitivo.joblib"
CSV_PATH = DATA_DIR / "dataset_limpio.csv"
OUTPUT_DIR = ROOT_DIR / "scripts"

# --- CONFIGURACIÓN DEL HOSPITAL ---
DAYS_IN_MONTH = 30
SLOT_MINUTES = 10
WORK_START_HOUR = 9
WORK_END_HOUR = 14
TOTAL_SLOTS = ((WORK_END_HOUR - WORK_START_HOUR) * 60) // SLOT_MINUTES
EMPTY_BREAK_SLOTS = {12, 13} 

FEATURES = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 
    'Handcap', 'SMS_received', 'Days_between', 'Appointment_Day_of_Week', 
    'Scheduled_Day_of_Week', 'Weekend', 'Appointment_Month', 
    'Scheduled_Month', 'Faltas_Previas', 'Citas_Previas', 'Ratio_Faltas', 
    'Gender_M', 'Scheduled_Time_of_Day_Evening', 'Scheduled_Time_of_Day_Morning'
]

def prepare_dataset_and_model():
    if not CSV_PATH.exists():
        print(f"❌ No encuentro el archivo {CSV_PATH}.")
        exit()
    df = pd.read_csv(CSV_PATH)
    columna_realidad = "No-show" if "No-show" in df.columns else "Falta_Real"
    if df[columna_realidad].dtype == "object":
        df["Falta_Real"] = df[columna_realidad].map({"Yes": True, "No": False, "1": True, "0": False, 1: True, 0: False})
    else:
        df["Falta_Real"] = df[columna_realidad] == 1

    modelo_ia = joblib.load(MODELO_JOBLIB_PATH)
    df["Prob_IA"] = modelo_ia.predict_proba(df[FEATURES])[:, 1]
    return df

def get_day_patients_pool(df, day_index, num_needed):
    rng = np.random.default_rng(day_index + 42)
    pool = df.sample(n=num_needed, random_state=day_index + 7).copy()
    patients = []
    for _, row in pool.iterrows():
        patients.append({
            "real_no_show": bool(row.get("Falta_Real", False)),
            "prob_no_show": float(row.get("Prob_IA", 0.2))
        })
    return patients

def create_schedule(day_patients_pool, smart_overbooking=False, umbral_riesgo=0.40):
    agenda = {slot: [] for slot in range(TOTAL_SLOTS)}
    pool_idx = 0
    overbookings_hechos = 0
    skip_slots = set(EMPTY_BREAK_SLOTS)
    
    # Asignación Base
    for slot in range(TOTAL_SLOTS):
        if slot in skip_slots: continue
        if pool_idx < len(day_patients_pool):
            agenda[slot].append(day_patients_pool[pool_idx])
            pool_idx += 1

    # Overbooking IA
    if smart_overbooking:
        for slot in range(TOTAL_SLOTS):
            if slot in skip_slots: continue
            if agenda[slot]:
                titular = agenda[slot][0]
                if titular["prob_no_show"] > umbral_riesgo and pool_idx < len(day_patients_pool):
                    agenda[slot].append(day_patients_pool[pool_idx])
                    pool_idx += 1
                    overbookings_hechos += 1
    return agenda, overbookings_hechos

def simulate_day_discrete_events(agenda):
    """
    Motor de Eventos Discretos: Simula la sala de espera minuto a minuto 
    aplicando la distribución Johnson SU para el desfase de llegadas.
    """
    slots_status = np.full(TOTAL_SLOTS, 3, dtype=int)
    pacientes_del_dia = []
    
    # 1. Generar la hora de llegada exacta de quienes NO faltan
    for slot in range(TOTAL_SLOTS):
        for p in agenda[slot]:
            if not p["real_no_show"]:
                # Parámetros del paper médico para la Johnson SU
                desfase = johnsonsu.rvs(a=-0.596, b=1.630, loc=-1.757, scale=24.270, size=1)[0]
                # Limitamos para que nadie llegue 2 horas antes ni 3 horas tarde
                desfase = np.clip(desfase, -45, 60) 
                
                minuto_citado = slot * SLOT_MINUTES
                minuto_llegada = minuto_citado + desfase
                
                # Guardamos la ficha del paciente
                pacientes_del_dia.append({
                    "slot_citado": slot,
                    "minuto_citado": minuto_citado,
                    "minuto_llegada": max(0, minuto_llegada) # No pueden llegar antes de abrir (min 0)
                })

    # Ordenamos a todos por el minuto en el que entran por la puerta de la clínica
    pacientes_del_dia.sort(key=lambda x: x["minuto_llegada"])
    
    sala_de_espera = []
    idx_llegadas = 0
    atendidos = 0
    solapamientos = 0
    max_retraso_dia = 0
    descansos_pendientes = 2
    
    # 2. El reloj del médico (Avanza slot a slot, 10 mins por paciente)
    for slot_actual in range(TOTAL_SLOTS):
        minuto_actual_reloj = slot_actual * SLOT_MINUTES
        
        # Ingresan a la sala de espera los que hayan llegado hasta este minuto
        while idx_llegadas < len(pacientes_del_dia) and pacientes_del_dia[idx_llegadas]["minuto_llegada"] <= minuto_actual_reloj + 9.99:
            sala_de_espera.append(pacientes_del_dia[idx_llegadas])
            idx_llegadas += 1

        # Detectar cuellos de botella reales en la sala física
        if len(sala_de_espera) > 1:
            solapamientos += 1

        # Lógica de Descanso Flexible
        if descansos_pendientes > 0 and slot_actual >= 11:
            if len(sala_de_espera) == 0 or slot_actual >= 16:
                slots_status[slot_actual] = 4 # Médico toma café
                descansos_pendientes -= 1
                continue
                
        # Lógica de Atención: Prioridad por Hora de Cita (El que antes tuviera la cita pasa primero)
        if len(sala_de_espera) > 0:
            # Ordenamos la sala por la hora a la que estaban CITADOS, no a la que llegaron
            sala_de_espera.sort(key=lambda x: x["slot_citado"])
            paciente_atendido = sala_de_espera.pop(0)
            
            # El retraso lo sufre el paciente desde que llega o desde su hora de cita (lo que sea mayor)
            tiempo_esperando = minuto_actual_reloj - max(paciente_atendido["minuto_citado"], paciente_atendido["minuto_llegada"])
            tiempo_esperando = max(0, tiempo_esperando)
            
            if tiempo_esperando > max_retraso_dia:
                max_retraso_dia = tiempo_esperando
                
            if tiempo_esperando < 5: 
                slots_status[slot_actual] = 1 # A tiempo (tolerancia < 5 min)
            else:
                slots_status[slot_actual] = 2 # Retrasado
                
            atendidos += 1
        else:
            # Si la sala está vacía
            if len(agenda[slot_actual]) > 0:
                slots_status[slot_actual] = 0 # No-show absoluto o viene hiper tarde
            else:
                slots_status[slot_actual] = 3 # Hueco vacío de agenda

    return slots_status, atendidos, solapamientos, max_retraso_dia

def simulate_month(df, overbooking_mode=False, umbral_riesgo=0.40):
    month_matrix = np.zeros((DAYS_IN_MONTH, TOTAL_SLOTS), dtype=int)
    stats = {"atendidos": 0, "solapamientos": 0, "overbookings": 0, "huecos_vacios": 0, "max_retraso_mes": 0}

    for day in range(DAYS_IN_MONTH):
        pool = get_day_patients_pool(df, day, num_needed=60)
        agenda, overbookings_dia = create_schedule(pool, smart_overbooking=overbooking_mode, umbral_riesgo=umbral_riesgo)
        day_matrix, atendidos_dia, solapamientos_dia, max_retraso_dia = simulate_day_discrete_events(agenda)
        
        month_matrix[day] = day_matrix
        stats["atendidos"] += atendidos_dia
        stats["solapamientos"] += solapamientos_dia
        stats["overbookings"] += overbookings_dia
        stats["huecos_vacios"] += int(np.sum((day_matrix == 0) | (day_matrix == 3)))
        
        if max_retraso_dia > stats["max_retraso_mes"]:
            stats["max_retraso_mes"] = max_retraso_dia

    return month_matrix, stats

def format_plot(ax, title):
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Slots de 10 min")
    ax.set_ylabel("Día del mes")
    slot_labels = [f"{WORK_START_HOUR + (i * SLOT_MINUTES // 60)}:{(i * SLOT_MINUTES) % 60:02d}" for i in range(TOTAL_SLOTS)]
    ax.set_xticks(np.arange(TOTAL_SLOTS))
    ax.set_xticklabels(slot_labels, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(DAYS_IN_MONTH))
    ax.set_yticklabels([str(d + 1) for d in range(DAYS_IN_MONTH)])

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#d3d3d3", markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#2ecc71", markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#f39c12", markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#3498db", markersize=12),
    ]
    legend_labels = ["No-show/Vacío", "A tiempo", "Retrasado", "Descanso Flexible"]
    ax.legend(legend_handles, legend_labels, loc="upper right")

def save_matrix_plot(matrix, title, path):
    cmap = plt.matplotlib.colors.ListedColormap(["#d3d3d3", "#2ecc71", "#f39c12", "#ffffff", "#3498db"])
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    format_plot(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def save_matrix_gif(matrix, title, path):
    frames = []
    cmap = plt.matplotlib.colors.ListedColormap(["#d3d3d3", "#2ecc71", "#f39c12", "#ffffff", "#3498db"])
    print(f"🎬 Renderizando GIF para {path.name}...")
    for day in range(1, DAYS_IN_MONTH + 1):
        temp_matrix = matrix.copy()
        if day < DAYS_IN_MONTH:
            temp_matrix[day:] = 3
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.imshow(temp_matrix, cmap=cmap, vmin=0, vmax=4, aspect="auto")
        format_plot(ax, f"{title} (Día {day})")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
    for _ in range(8): frames.append(frames[-1])
    frames[0].save(path, format='GIF', append_images=frames[1:], save_all=True, duration=250, loop=0)

def main():
    df = prepare_dataset_and_model()

    print("--- EVALUANDO ESCENARIO TRADICIONAL (MUNDO REAL SIN IA) ---")
    matrix_trad, stats_trad = simulate_month(df, overbooking_mode=False)
    print(f"📊 Tradicional -> Atendidos: {stats_trad['atendidos']} | Peor Retraso: {stats_trad['max_retraso_mes']:.1f} min")
    print("-" * 75)

    umbrales_a_probar = [0.30, 0.40, 0.45, 0.50, 0.60]
    mejor_umbral, max_score, mejor_matrix, mejor_stats = None, -float('inf'), None, None

    print("🔬 PROBANDO IA CON EVENTOS DISCRETOS Y JOHNSON S_U:")
    for umbral in umbrales_a_probar:
        matrix_ia, stats_ia = simulate_month(df, overbooking_mode=True, umbral_riesgo=umbral)
        extra = stats_ia['atendidos'] - stats_trad['atendidos']
        
        # Penalización severa para retrasos inaceptables
        penalizacion_tiempo = 0
        if stats_ia['max_retraso_mes'] > 45:
            penalizacion_tiempo = (stats_ia['max_retraso_mes'] - 45) * 3

        score = (extra * 2.0) - (stats_ia['solapamientos'] * 0.5) - penalizacion_tiempo
        print(f"   🎯 Umbral > {umbral:.2f} | Atendidos: {stats_ia['atendidos']} (+{extra}) | Solapamientos: {stats_ia['solapamientos']} | Peor Retraso: {stats_ia['max_retraso_mes']:.1f} min | Score: {score:.1f}")

        if score > max_score:
            max_score = score
            mejor_umbral = umbral
            mejor_matrix = matrix_ia
            mejor_stats = stats_ia

    incremento_pacientes = mejor_stats['atendidos'] - stats_trad['atendidos']
    porcentaje_mejora = (incremento_pacientes / stats_trad['atendidos']) * 100

    print("=" * 75)
    print(f"🏆 CONCLUSIÓN ÓPTIMA: Umbral Recomendado > {mejor_umbral}")
    if porcentaje_mejora > 0:
        print(f"💰 IMPACTO DE NEGOCIO: Aumento del {porcentaje_mejora:.1f}% (+{incremento_pacientes} pacientes)")
        print(f"⏱️ COMPROMISO DE CALIDAD: El peor retraso del mes fue de {mejor_stats['max_retraso_mes']:.1f} minutos, habiendo absorbido el caos humano (llegadas tempranas y tardías).")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_matrix_plot(matrix_trad, "Agenda Tradicional (Caos Real)", OUTPUT_DIR / "mc_4_tradicional.png")
    save_matrix_plot(mejor_matrix, f"Agenda Inteligente (Absorción de Caos)", OUTPUT_DIR / "mc_4_inteligente.png")
    save_matrix_gif(matrix_trad, "Evolución Tradicional", OUTPUT_DIR / "mc_4_animacion_trad.gif")
    save_matrix_gif(mejor_matrix, "Evolución Inteligente", OUTPUT_DIR / "mc_4_animacion_ia.gif")
    
    print(f"\n✅ Simulación completada. Renderizados guardados en 'scripts'.")

if __name__ == "__main__":
    main()