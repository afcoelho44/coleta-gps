"""
Análise do posicionamento: erro de precisão e erro de distância, sem e com filtro de Kalman.

Uso:
    python analisar.py coleta_*.csv [--aquecimento 15] [--q 0.5] [--saida resultados]

Entrada: um ou mais CSVs exportados pela página de coleta (colunas participante, local,
distancia_m, ambiente, lat, lng, accuracy, ...). CSVs antigos sem distancia_m entram só
no cálculo de precisão.

O que é calculado
-----------------
1. Precisão (todas as capturas): o quanto a posição oscila com o celular parado.
   SD_2D = raiz da média do quadrado da distância de cada leitura até a média da captura.

2. Erro de distância (capturas a 5, 10 e 20 m):
   - A coordenada do objeto é definida como o professor faria: a média das leituras brutas
     das capturas a 0 m (todas as repetições) do mesmo participante, dispositivo, ambiente e ponto.
   - Para cada leitura: d_gps = Haversine(leitura, objeto); erro = d_gps - distância da fita.
   - Métricas por grupo: viés (média do erro com sinal), ME (média do erro absoluto),
     SD (desvio-padrão do erro) e RMSE.

Os dois cálculos são feitos para o fluxo bruto e para o fluxo filtrado pelo Kalman.
O Kalman é aplicado por captura (estado reiniciado a cada sequência), em metros, com modelo de
velocidade constante por eixo e R = accuracy² informada pelo navegador.
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

R_TERRA = 6371008.8


def haversine(lat1, lng1, lat2, lng2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lng2) - np.asarray(lng1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_TERRA * np.arcsin(np.sqrt(a))


def carregar(arquivos):
    partes = []
    for f in arquivos:
        d = pd.read_csv(f)
        d['arquivo'] = os.path.basename(f)
        partes.append(d)
    d = pd.concat(partes, ignore_index=True)
    d = d.drop_duplicates(subset=['captura_id', 'leitura_idx'])
    for col, padrao in [('participante', 'sem_id'), ('distancia_m', np.nan), ('fix_antigo', None)]:
        if col not in d.columns:
            d[col] = padrao
    d['participante'] = d['participante'].fillna('sem_id').astype(str)
    d['local'] = d['local'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    d['dispositivo_curto'] = np.where(d['dispositivo'].str.contains('iPhone', na=False), 'iPhone',
                             np.where(d['dispositivo'].str.contains('Android', na=False), 'Android', 'outro'))
    # leituras em cache (de antes de apertar Capturar) não entram em nada
    ts = pd.to_datetime(d['gps_timestamp'], utc=True)
    ini = pd.to_datetime(d['inicio'], utc=True)
    antigo = (ts < ini) | (d['fix_antigo'].astype(str).str.lower() == 'true')
    d = d[~antigo].copy()
    d['t_gps_s'] = ts[~antigo].astype('int64') / 1e9
    return d.sort_values(['captura_id', 't_gps_s', 'leitura_idx']).reset_index(drop=True)


def kalman_captura(g, q):
    """Kalman 2D (eixos independentes, estado [posição, velocidade]) em metros locais."""
    lat0, lng0 = g['lat'].iloc[0], g['lng'].iloc[0]
    m_lat = math.pi / 180 * R_TERRA
    m_lng = m_lat * math.cos(math.radians(lat0))
    xs = (g['lng'].to_numpy() - lng0) * m_lng
    ys = (g['lat'].to_numpy() - lat0) * m_lat
    t = g['t_gps_s'].to_numpy()
    r = np.maximum(g['accuracy'].to_numpy(), 1.0) ** 2

    saida = np.zeros((len(g), 2))
    for eixo, z in enumerate((xs, ys)):
        x = np.array([z[0], 0.0])
        P = np.diag([r[0], 4.0])          # incerteza inicial: accuracy² e (2 m/s)²
        saida[0, eixo] = x[0]
        for k in range(1, len(z)):
            dt = max(t[k] - t[k - 1], 1e-3)
            F = np.array([[1, dt], [0, 1]])
            Q = q ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
            x = F @ x
            P = F @ P @ F.T + Q
            S = P[0, 0] + r[k]
            K = P[:, 0] / S
            x = x + K * (z[k] - x[0])
            P = P - np.outer(K, P[0, :])
            saida[k, eixo] = x[0]
    return lat0 + saida[:, 1] / m_lat, lng0 + saida[:, 0] / m_lng


def aplicar_kalman(d, q):
    d['lat_k'] = np.nan
    d['lng_k'] = np.nan
    for _, g in d.groupby('captura_id', sort=False):
        la, ln = kalman_captura(g, q)
        d.loc[g.index, 'lat_k'] = la
        d.loc[g.index, 'lng_k'] = ln
    return d


FLUXOS = {'bruto': ('lat', 'lng'), 'kalman': ('lat_k', 'lng_k')}
CHAVE_OBJ = ['participante', 'dispositivo_curto', 'ambiente', 'local']


def metricas_erro(e):
    e = np.asarray(e, dtype=float)
    return pd.Series({'n': len(e), 'vies_m': e.mean(), 'ME_m': np.abs(e).mean(),
                      'SD_m': e.std(ddof=1) if len(e) > 1 else np.nan,
                      'RMSE_m': math.sqrt((e ** 2).mean())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    ap.add_argument('--aquecimento', type=float, default=15, help='segundos descartados no início de cada captura')
    ap.add_argument('--q', type=float, default=0.5, help='ruído de aceleração do Kalman (m/s²)')
    ap.add_argument('--saida', default='resultados')
    a = ap.parse_args()

    arquivos = sorted({f for padrao in a.arquivos for f in glob.glob(padrao)})
    d = aplicar_kalman(carregar(arquivos), a.q)
    d['avaliar'] = d['t_ms'] >= a.aquecimento * 1000   # o filtro roda desde o início; a avaliação começa depois
    os.makedirs(a.saida, exist_ok=True)

    # ---------- coordenada do objeto: média das leituras brutas a 0 m ----------
    zero = d[(d['distancia_m'] == 0) & d['avaliar']]
    objetos = zero.groupby(CHAVE_OBJ).agg(obj_lat=('lat', 'mean'), obj_lng=('lng', 'mean'),
                                          n_obj=('lat', 'size')).reset_index()
    d = d.merge(objetos, on=CHAVE_OBJ, how='left')

    linhas_cap, linhas_erro = [], []
    for fluxo, (cl, cg) in FLUXOS.items():
        ok = d[d['avaliar']]
        # precisão por captura
        for cid, g in ok.groupby('captura_id'):
            la0, ln0 = g[cl].mean(), g[cg].mean()
            r = haversine(la0, ln0, g[cl].to_numpy(), g[cg].to_numpy())
            linhas_cap.append({'fluxo': fluxo, 'captura_id': cid, **g.iloc[0][CHAVE_OBJ + ['distancia_m', 'repeticao']].to_dict(),
                               'n': len(g), 'accuracy_mediana_m': g['accuracy'].median(),
                               'SD_2D_m': math.sqrt((r ** 2).mean()), 'max_afastamento_m': r.max()})
        # erro de distância por leitura
        dist = ok[(ok['distancia_m'] > 0) & ok['obj_lat'].notna()].copy()
        if len(dist):
            dist['d_gps_m'] = haversine(dist['obj_lat'], dist['obj_lng'], dist[cl], dist[cg])
            dist['erro_m'] = dist['d_gps_m'] - dist['distancia_m']
            dist['fluxo'] = fluxo
            linhas_erro.append(dist[['fluxo', 'captura_id'] + CHAVE_OBJ + ['distancia_m', 'repeticao', 't_ms',
                                     'accuracy', 'd_gps_m', 'erro_m']])

    caps = pd.DataFrame(linhas_cap)
    caps.to_csv(os.path.join(a.saida, 'precisao_por_captura.csv'), index=False)
    prec = (caps.groupby(['ambiente', 'fluxo'])
                .agg(capturas=('captura_id', 'size'), SD_2D_medio_m=('SD_2D_m', 'mean'),
                     accuracy_mediana_m=('accuracy_mediana_m', 'median'))
                .reset_index())
    prec.to_csv(os.path.join(a.saida, 'precisao_resumo.csv'), index=False)

    pd.set_option('display.width', 200)
    print(f'\n{len(arquivos)} arquivos, {d["captura_id"].nunique()} capturas, '
          f'aquecimento {a.aquecimento:g} s, Kalman q = {a.q:g} m/s²')
    print('\n== Precisão (SD 2D com o celular parado) ==')
    print(prec.round(2).to_string(index=False))

    if linhas_erro:
        erros = pd.concat(linhas_erro, ignore_index=True)
        erros.to_csv(os.path.join(a.saida, 'erro_distancia_por_leitura.csv'), index=False)
        resumo = (erros.groupby(['ambiente', 'distancia_m', 'fluxo'])['erro_m']
                       .apply(metricas_erro).unstack().reset_index())
        resumo.to_csv(os.path.join(a.saida, 'erro_distancia_resumo.csv'), index=False)
        print('\n== Erro de distância (d_gps - d_fita) ==')
        print(resumo.round(2).to_string(index=False))
        sem_obj = d[(d['distancia_m'] > 0) & d['obj_lat'].isna()].groupby(CHAVE_OBJ).size()
        if len(sem_obj):
            print('\nAtenção: capturas a 5/10/20 m sem captura a 0 m correspondente (ficaram de fora):')
            print(sem_obj.to_string())
    else:
        print('\nNenhuma captura a 5/10/20 m com objeto a 0 m: erro de distância não calculado.')

    d.to_csv(os.path.join(a.saida, 'leituras_processadas.csv'), index=False)
    print(f'\nArquivos em {a.saida}/')


if __name__ == '__main__':
    main()
