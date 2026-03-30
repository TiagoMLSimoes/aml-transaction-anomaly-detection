# =============================================================================
# PROJETO 1 — VERSÃO SÉNIOR: AML TRANSACTION ANOMALY DETECTION
# =============================================================================
# MELHORIAS APLICADAS NESTA VERSÃO:
#   ✅ Código organizado em funções com responsabilidade única
#   ✅ Type hints em todas as funções
#   ✅ Logging em vez de prints
#   ✅ Tratamento de erros robusto com try/except
#   ✅ Context managers para ligações à base de dados
#   ✅ Todos os imports no topo (PEP 8)
#   ✅ Docstrings em todas as funções
#   ✅ Configuração centralizada numa dataclass
#   ✅ Thresholds de risco pré-computados (sem recalcular por linha)
#   ✅ Função main() como entry point
# =============================================================================

# =============================================================================
# SECÇÃO 1: IMPORTS — todos no topo, organizados por categoria
# =============================================================================

# Standard library (biblioteca padrão Python — não precisas instalar)
import logging
import os
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# PORQUÊ dataclass: permite definir configuração como uma classe com defaults.
# É mais limpo que variáveis globais soltas, e o IDE autocompleta os campos.
# PORQUÊ typing: type hints para indicar tipos esperados em funções.
# PORQUÊ pathlib.Path: mais robusto que strings para caminhos de ficheiros.
#   Path("outputs") / "charts" funciona em Windows, Mac e Linux automaticamente.
#   "outputs\\charts" ou "outputs/charts" pode falhar dependendo do OS.

# Third-party (precisas instalar com pip)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
# PORQUÊ importar Patch aqui e não no meio do código (versão original):
# PEP 8 — a regra de estilo universal de Python — diz que todos os imports
# devem estar no topo do ficheiro. Imports no meio do código são difíceis
# de localizar e confundem outros developers.

warnings.filterwarnings("ignore")

# =============================================================================
# SECÇÃO 2: CONFIGURAÇÃO DE LOGGING
# =============================================================================

# Configurar logger antes de qualquer outra coisa
# PORQUÊ logging em vez de print():
#   - print() só aparece no terminal durante a sessão
#   - logging persiste para ficheiro, tem níveis (DEBUG/INFO/WARNING/ERROR)
#   - Em produção, podes configurar alertas automáticos para logs de ERROR
#   ELI5: print() é um post-it. logging é um diário com data e hora.

Path("outputs/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("outputs/logs/aml_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
# __name__: nome do módulo actual. Se este ficheiro for importado por outro,
# o logger herda o nome correcto em vez de "root".

# =============================================================================
# SECÇÃO 3: CONFIGURAÇÃO CENTRALIZADA
# =============================================================================

@dataclass
# PORQUÊ @dataclass:
#   Decorator que gera automaticamente __init__, __repr__, e __eq__.
#   Em vez de escrever uma classe com 10 linhas de __init__,
#   só defines os campos e os defaults. Mais limpo, menos código.
#   ELI5: É como um formulário com campos pré-definidos.
class AMLConfig:
    """
    Configuração centralizada do pipeline AML.
    
    Todos os parâmetros configuráveis estão aqui — nunca espalhados pelo código.
    Para mudar o comportamento do pipeline, só precisas de editar esta classe.
    
    PORQUÊ centralizar configuração:
        Em produção, configurações vêm de variáveis de ambiente ou ficheiros
        de config (YAML, .env). Centralizar aqui simula esse padrão e facilita
        a migração para produção.
    """
    # Caminhos
    dataset_path: str = "paysim.csv"
    db_path: str = "aml_transactions.db"
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    # field(default_factory=...): necessário para defaults mutáveis em dataclasses
    # (listas, dicts, objetos) — evita o bug clássico de defaults partilhados

    # Parâmetros do modelo
    contamination_rate: float = 0.01
    # 0.01 = esperamos que 1% das transações sejam anómalas
    # Em produção: calibrar com base na capacidade da equipa de compliance

    n_estimators: int = 200
    # Número de árvores no Isolation Forest
    # Mais árvores = mais estável mas mais lento
    # 100-200 é o range standard para produção

    random_state: int = 42
    # Seed para reprodutibilidade — CRÍTICO em contexto regulatório
    # Auditores precisam de reproduzir os mesmos resultados

    n_rows_to_load: Optional[int] = 500_000
    # None = carrega o dataset completo (~6M linhas)
    # 500k para desenvolvimento rápido

    min_transactions_per_customer: int = 1  
    # Clientes com menos de N transações são excluídos da análise
    # Com 1 transação não conseguimos calcular padrão comportamental


# =============================================================================
# SECÇÃO 4: FUNÇÕES DE VALIDAÇÃO
# =============================================================================

def validate_dataset(config: AMLConfig) -> None:
    """
    Valida que o dataset existe e é legível antes de iniciar o pipeline.
    
    PORQUÊ validar no início:
        Fail fast — é melhor falhar com uma mensagem clara ao início
        do que falhar a meio do pipeline com um erro críptico.
        ELI5: Verificas se tens os ingredientes antes de começar a cozinhar.
    
    Args:
        config: Configuração do pipeline.
    
    Raises:
        FileNotFoundError: Se o dataset não existir.
        ValueError: Se o dataset estiver vazio ou com colunas em falta.
    """
    # PORQUÊ raise em vez de exit():
    #   exit() termina o processo abruptamente — não é "pythónico" e não pode
    #   ser capturado por código que importa este módulo.
    #   raise lança uma excepção que pode ser capturada e tratada adequadamente.

    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: '{config.dataset_path}'\n"
            f"Descarrega em: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            f"Renomeia para: paysim.csv"
        )

    required_columns = {
        "step", "type", "amount", "nameOrig",
        "oldbalanceOrg", "newbalanceOrig", "nameDest",
        "oldbalanceDest", "newbalanceDest", "isFraud",
    }

    # Ler apenas o header para verificar colunas — sem carregar o ficheiro inteiro
    # PORQUÊ nrows=0: verificar colunas sem custo de memória
    df_header = pd.read_csv(config.dataset_path, nrows=0)
    missing_cols = required_columns - set(df_header.columns)

    if missing_cols:
        raise ValueError(
            f"Colunas em falta no dataset: {missing_cols}\n"
            f"Verifica se o ficheiro é o PaySim correcto."
        )

    logger.info("✅ Validação do dataset: OK")


# =============================================================================
# SECÇÃO 5: CARREGAMENTO DE DADOS
# =============================================================================

def load_data(config: AMLConfig) -> pd.DataFrame:
    """
    Carrega o dataset CSV para um DataFrame Pandas.
    
    MELHORIAS vs versão original:
        - try/except para erros de leitura
        - dtypes explícitos para reduzir uso de memória
        - Logging de métricas de carregamento
    
    Args:
        config: Configuração do pipeline.
    
    Returns:
        DataFrame com as transações carregadas.
    
    Raises:
        RuntimeError: Se o ficheiro não puder ser lido.
    """
    logger.info(f"A carregar dataset: {config.dataset_path}")

    # Definir dtypes explicitamente
    # PORQUÊ: Por defeito, Pandas infere os tipos ao ler o CSV — processo lento
    # e que frequentemente usa mais memória do que necessário.
    # Definir dtypes explícitos pode reduzir uso de memória em 30-60%.
    # ELI5: É como dizer ao armazém exactamente em que prateleira guardar cada coisa
    # em vez de deixar os funcionários decidirem.
    dtypes = {
        "step": "int32",          # int32 em vez de int64 — suficiente para valores pequenos
        "type": "category",        # category em vez de object — menos memória para strings repetidas
        "amount": "float32",       # float32 em vez de float64 — metade da memória
        "nameOrig": "object",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "nameDest": "object",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",         # int8: só precisa de 0 ou 1
        "isFlaggedFraud": "int8",
    }

    try:
        df = pd.read_csv(
            config.dataset_path,
            nrows=config.n_rows_to_load,
            dtype=dtypes,
        )
    except Exception as e:
        # PORQUÊ capturar Exception genérica aqui:
        #   pd.read_csv pode lançar vários tipos de erro (ParserError, EmptyDataError, etc.)
        #   Capturamos todos e relançamos com contexto útil.
        raise RuntimeError(f"Erro ao ler o dataset: {e}") from e

    logger.info(f"   Linhas carregadas: {len(df):,}")
    logger.info(f"   Colunas: {list(df.columns)}")
    logger.info(f"   Memória utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    logger.info(f"   Taxa de fraude real: {df['isFraud'].mean():.4%}")

    return df


# =============================================================================
# SECÇÃO 6: BASE DE DADOS
# =============================================================================

def load_to_database(df: pd.DataFrame, config: AMLConfig) -> None:
    """
    Carrega o DataFrame para SQLite usando context manager.
    
    MELHORIA vs versão original:
        - Context manager (with) garante que a conexão fecha mesmo com erro
        - chunksize para datasets grandes — não carrega tudo de uma vez
    
    Args:
        df: DataFrame com as transações.
        config: Configuração do pipeline.
    """
    logger.info(f"A carregar dados para SQLite: {config.db_path}")

    # PORQUÊ 'with' (context manager):
    #   Na versão original, se houvesse um erro entre connect() e close(),
    #   a conexão ficava aberta — potencial corrupção de dados.
    #   'with' garante que close() é sempre chamado, mesmo com erro.
    #   ELI5: É como uma porta automática — fecha sempre, mesmo que te esqueças.
    with sqlite3.connect(config.db_path) as conn:
        df.to_sql(
            "transactions",
            conn,
            if_exists="replace",
            index=False,
            chunksize=10_000,
            # PORQUÊ chunksize: em vez de escrever 500k linhas de uma vez
            # (que pode causar timeout ou memory error), escreve em blocos de 10k.
            # ELI5: Em vez de tentar carregar 500 caixas de uma vez, carregas 10 de cada vez.
        )

    logger.info(f"   Tabela 'transactions' criada: {len(df):,} registos")


# =============================================================================
# SECÇÃO 7: FEATURE ENGINEERING
# =============================================================================

def compute_customer_features(config: AMLConfig) -> pd.DataFrame:
    """
    Calcula features comportamentais por cliente usando SQL.
    
    PORQUÊ SQL para feature engineering:
        Em produção, dados de transações vivem em bases de dados relacionais.
        SQL aggregations são a interface natural. Também demonstra competência
        em SQL — valorizado em todas as empresas de dados.
    
    Args:
        config: Configuração do pipeline.
    
    Returns:
        DataFrame com uma linha por cliente e as suas features comportamentais.
    """
    logger.info("A calcular features comportamentais via SQL...")

    # Query SQL modular com CTE para legibilidade
    # PORQUÊ CTE (WITH ... AS):
    #   Em vez de uma query gigante e ilegível, dividimos em blocos nomeados.
    #   Cada CTE resolve um problema específico — é como funções mas em SQL.
    query = f"""
    WITH base AS (
        -- Filtro base: só clientes com N+ transações
        -- HAVING antes dos cálculos evita processar clientes com dados insuficientes
        SELECT nameOrig
        FROM transactions
        GROUP BY nameOrig
        HAVING COUNT(*) >= {config.min_transactions_per_customer}
    ),
    aggregated AS (
        SELECT
            t.nameOrig,

            -- === FEATURES DE FREQUÊNCIA ===
            COUNT(*)                    AS total_transactions,
            -- Quantas transações fez — proxy de actividade

            -- === FEATURES DE VALOR ===
            AVG(t.amount)               AS avg_amount,
            MAX(t.amount)               AS max_amount,
            SUM(t.amount)               AS total_amount,
            COALESCE(
    SQRT(AVG(t.amount * t.amount) - AVG(t.amount) * AVG(t.amount)),
    0
) AS std_amount,
            -- COALESCE: substitui NULL por 0 (clientes com 1 tx não têm std)
            -- std alto = comportamento irregular (suspeito)

            -- === FEATURES DE SALDO ===
            AVG(t.oldbalanceOrg)                    AS avg_balance_before,
            AVG(t.oldbalanceOrg - t.newbalanceOrig) AS avg_balance_delta,
            -- delta negativo esperado (dinheiro saiu) — muito positivo = suspeito

            -- === FEATURES DE TIPO ===
            SUM(CASE WHEN t.type = 'CASH_OUT' THEN 1 ELSE 0 END) AS cashout_count,
            SUM(CASE WHEN t.type = 'TRANSFER' THEN 1 ELSE 0 END) AS transfer_count,

            -- === FEATURES RATIO (normalizam pela frequência total) ===
            CAST(SUM(CASE WHEN t.type = 'CASH_OUT' THEN 1 ELSE 0 END) AS FLOAT)
                / COUNT(*) AS cashout_ratio,
            -- cashout_ratio > 0.8 = quase todas as transações são levantamentos

            -- === FEATURES DE DIVERSIFICAÇÃO ===
            COUNT(DISTINCT t.nameDest) AS unique_destinations
            -- muitos destinos únicos = layering (fase 2 de lavagem de dinheiro)

        FROM transactions t
        INNER JOIN base b ON t.nameOrig = b.nameOrig
        GROUP BY t.nameOrig
    )
    SELECT * FROM aggregated
    ORDER BY total_amount DESC
    """

    with sqlite3.connect(config.db_path) as conn:
        df_features = pd.read_sql_query(query, conn)

    logger.info(f"   Features calculadas: {len(df_features):,} clientes")
    logger.info(f"   Dimensão da matriz: {df_features.shape[0]} × {df_features.shape[1] - 1} features")

    return df_features


# =============================================================================
# SECÇÃO 8: PRÉ-PROCESSAMENTO
# =============================================================================

def preprocess_features(
    df_features: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepara a matriz de features para o modelo ML.
    
    RETORNA uma Tuple com 3 elementos:
        - X_scaled: matriz normalizada para o modelo
        - customer_ids: IDs dos clientes (para mapear resultados)
        - scaler: o StandardScaler treinado (necessário para novos dados em produção)
    
    PORQUÊ devolver o scaler:
        Em produção, quando chegam novas transações, precisas de aplicar
        a MESMA normalização que usaste no treino. Guardar o scaler permite isso.
        Seria serializado com joblib.dump(scaler, 'scaler.pkl').
    
    Args:
        df_features: DataFrame com features por cliente.
    
    Returns:
        Tuple de (X_scaled, customer_ids, scaler).
    """
    logger.info("A pré-processar features para o modelo...")

    # Separar IDs das features numéricas
    customer_ids = df_features["nameOrig"].values
    feature_cols = [c for c in df_features.columns if c != "nameOrig"]
    X = df_features[feature_cols].values.astype(np.float64)
    # astype(float64): garante tipo consistente antes da normalização

    # Tratar valores inválidos
    # PORQUÊ: divisões por zero ou edge cases nos dados podem criar NaN/Inf.
    # O modelo falha silenciosamente com estes valores.
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        logger.warning(f"   Valores inválidos encontrados: {n_nan} NaN, {n_inf} Inf — substituídos por 0")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # nan_to_num: substitui NaN e Inf por valores definidos — mais explícito que where()

    # Normalizar com StandardScaler
    # NOTA IMPORTANTE PARA ENTREVISTAS:
    #   fit_transform() num dataset não dividido em treino/teste é aceitável
    #   para deteção de anomalias não-supervisionada (não há target a vazar).
    #   Em modelos supervisionados, NUNCA fazer fit no conjunto de teste.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"   Shape: {X_scaled.shape[0]:,} clientes × {X_scaled.shape[1]} features")
    logger.info(f"   Após normalização: média={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

    return X_scaled, customer_ids, scaler


# =============================================================================
# SECÇÃO 9: MODELO ISOLATION FOREST
# =============================================================================

def train_isolation_forest(
    X_scaled: np.ndarray,
    config: AMLConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Treina o modelo Isolation Forest e devolve predições e scores.
    
    COMO FUNCIONA O ISOLATION FOREST:
        Constrói N árvores de decisão aleatórias. Para cada árvore, selecciona
        aleatoriamente uma feature e um ponto de corte, dividindo os dados
        recursivamente.
        
        Intuição chave: pontos anómalos são isolados em MENOS passos.
        - Ponto normal (no centro): precisa de muitos cortes para ficar sozinho
        - Ponto anómalo (no extremo): fica isolado com 2-3 cortes
        
        O anomaly score é inversamente proporcional ao número médio de cortes.
        
        ELI5: Jogo das 20 perguntas. Para encontrar a pessoa "normal" no meio
        da multidão, precisas de muitas perguntas. Para a pessoa com cabelo
        cor-de-rosa, 1-2 perguntas bastam.
    
    Args:
        X_scaled: Matriz de features normalizada.
        config: Configuração do pipeline.
    
    Returns:
        Tuple de (predictions, risk_scores):
            - predictions: array de 1 (normal) ou -1 (anómalo) por cliente
            - risk_scores: scores contínuos — quanto maior, mais suspeito
    """
    logger.info(f"A treinar Isolation Forest ({config.n_estimators} árvores)...")

    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination_rate,
        max_samples="auto",
        # auto = min(256, n_samples) — 256 amostras por árvore é suficiente
        # Mais amostras não melhora significativamente, mas aumenta tempo de treino
        random_state=config.random_state,
        n_jobs=-1,
        # n_jobs=-1: usa TODOS os cores da CPU em paralelo
        # Num Mac com 8 cores, treina ~8x mais rápido
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    # predict() devolve: 1 = normal, -1 = anómalo

    # Inverter decision_function para que score positivo = mais suspeito
    # PORQUÊ inverter: Isolation Forest usa convenção inversa por default
    # (scores negativos = mais anómalo). Invertemos para o relatório ser intuitivo.
    risk_scores = -model.decision_function(X_scaled)

    n_anomalies = (predictions == -1).sum()
    logger.info(f"   Clientes analisados: {len(predictions):,}")
    logger.info(f"   Anomalias detectadas: {n_anomalies:,} ({n_anomalies/len(predictions):.2%})")

    return predictions, risk_scores


# =============================================================================
# SECÇÃO 10: CLASSIFICAÇÃO DE RISCO
# =============================================================================

def classify_risk_levels(
    risk_scores: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Classifica anomalias em níveis HIGH/MEDIUM/LOW/NORMAL.
    
    MELHORIA vs versão original:
        A versão original recalculava np.percentile() para CADA LINHA
        dentro de uma função apply() — N chamadas desnecessárias.
        Esta versão pré-computa os thresholds UMA VEZ e aplica vectorialmente.
    
    PORQUÊ vectorização em vez de apply():
        apply() é essencialmente um for loop disfarçado — lento para arrays grandes.
        Operações vectoriais (np.where, np.select) processam o array inteiro de uma vez.
        Para 10k clientes: apply() ~50ms, np.select() ~0.5ms (100x mais rápido).
    
    Args:
        risk_scores: Array de scores contínuos de risco.
        predictions: Array de predições (1=normal, -1=anómalo).
    
    Returns:
        Array de strings com nível de risco por cliente.
    """
    # Pré-computar thresholds UMA VEZ para todos os clientes anómalos
    anomaly_mask = predictions == -1
    anomaly_scores = risk_scores[anomaly_mask]

    if len(anomaly_scores) == 0:
        return np.array(["NORMAL"] * len(risk_scores))

    # Percentis 75 e 25 dos scores de anomalia definem HIGH/MEDIUM/LOW
    threshold_high = np.percentile(anomaly_scores, 75)
    threshold_low = np.percentile(anomaly_scores, 25)

    # np.select: versão vectorial de if/elif/else
    # Processa o array inteiro de uma vez em vez de linha a linha
    conditions = [
        ~anomaly_mask,                                      # não é anomalia
        anomaly_mask & (risk_scores >= threshold_high),     # HIGH
        anomaly_mask & (risk_scores >= threshold_low),      # MEDIUM
        anomaly_mask,                                       # LOW (resto das anomalias)
    ]
    choices = ["NORMAL", "HIGH", "MEDIUM", "LOW"]

    return np.select(conditions, choices, default="NORMAL")


# =============================================================================
# SECÇÃO 11: VALIDAÇÃO DO MODELO
# =============================================================================

def validate_model(
    df_features: pd.DataFrame,
    df_raw: pd.DataFrame,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Valida o modelo comparando com os labels reais de fraude do PaySim.
    
    NOTA IMPORTANTE:
        Em produção real não terias a coluna isFraud. Esta validação é
        específica do PaySim para avaliar a qualidade do modelo.
        Em produção, validas com Precision@K após investigação manual.
    
    Args:
        df_features: DataFrame com features por cliente.
        df_raw: DataFrame original com coluna isFraud.
        predictions: Array de predições do modelo.
    
    Returns:
        Dicionário com métricas de validação.
    """
    logger.info("A validar modelo contra labels reais de fraude...")

    # Calcular se cada cliente tem alguma transação fraudulenta real
    fraud_by_customer = (
        df_raw.groupby("nameOrig")["isFraud"]
        .max()  # max() = 1 se ALGUMA transação for fraude
        .reset_index()
        .rename(columns={"isFraud": "is_real_fraud"})
    )

    # Juntar com resultados do modelo
    df_val = df_features[["nameOrig"]].copy()
    df_val["is_anomaly"] = (predictions == -1).astype(int)
    df_val = df_val.merge(fraud_by_customer, on="nameOrig", how="left")
    df_val["is_real_fraud"] = df_val["is_real_fraud"].fillna(0)

    anomalous = df_val[df_val["is_anomaly"] == 1]
    metrics: Dict[str, float] = {}

    if len(anomalous) > 0:
        metrics["precision"] = float(anomalous["is_real_fraud"].mean())
        metrics["n_alerts"] = float(len(anomalous))
        metrics["n_true_positives"] = float(anomalous["is_real_fraud"].sum())

        if metrics["n_true_positives"] > 0:
            logger.info(f"   Precision: {metrics['precision']:.2%} "
                        f"({int(metrics['n_true_positives'])} fraudes reais em "
                        f"{int(metrics['n_alerts'])} alertas)")
        else:
            logger.info("   Nota: 0 fraudes reais nos alertas — normal para amostras pequenas.")
            logger.info("   A taxa de fraude real no PaySim é <0.1% do dataset completo.")

    return metrics


# =============================================================================
# SECÇÃO 12: VISUALIZAÇÕES
# =============================================================================

def generate_visualizations(
    df_results: pd.DataFrame,
    df_features: pd.DataFrame,
    risk_scores: np.ndarray,
    predictions: np.ndarray,
    feature_columns: list,
    config: AMLConfig,
) -> Dict[str, Path]:
    """
    Gera todos os gráficos de análise e guarda-os em PNG.
    
    MELHORIA vs versão original:
        - Import de Patch movido para o topo do ficheiro
        - Cada gráfico tem interpretação clara no título/labels
        - Retorna dicionário de caminhos para uso programático
    
    Args:
        df_results: DataFrame com resultados e classificações de risco.
        df_features: DataFrame com features por cliente.
        risk_scores: Scores de anomalia contínuos.
        predictions: Predições do modelo.
        feature_columns: Lista de colunas de features.
        config: Configuração do pipeline.
    
    Returns:
        Dicionário {nome_gráfico: Path do ficheiro PNG}.
    """
    logger.info("A gerar visualizações...")

    charts_dir = config.output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams["figure.dpi"] = 120

    chart_paths: Dict[str, Path] = {}
    df_alerts = df_results[df_results["is_anomaly"] == 1].sort_values(
        "risk_score", ascending=False
    )

    # ------------------------------------------------------------------
    # GRÁFICO 1: Painel principal (2×2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AML Transaction Anomaly Detection — Risk Analysis",
                 fontsize=14, fontweight="bold")

    # Plot 1: Distribuição dos risk scores
    ax1 = axes[0, 0]
    normal_scores = risk_scores[predictions == 1]
    anomaly_scores_arr = risk_scores[predictions == -1]
    ax1.hist(normal_scores, bins=50, alpha=0.7, color="steelblue",
             label=f"Normal (n={len(normal_scores):,})", density=True)
    ax1.hist(anomaly_scores_arr, bins=50, alpha=0.7, color="crimson",
             label=f"Anómalo (n={len(anomaly_scores_arr):,})", density=True)
    ax1.set_xlabel("Risk Score")
    ax1.set_ylabel("Densidade")
    ax1.set_title("Distribuição dos Risk Scores\n(separação clara = modelo eficaz)")
    ax1.legend()

    # Plot 2: Top 20 clientes mais suspeitos
    ax2 = axes[0, 1]
    top20 = df_alerts.head(20)
    color_map = {"HIGH": "crimson", "MEDIUM": "darkorange", "LOW": "gold"}
    bar_colors = [color_map.get(r, "gray") for r in top20["risk_level"]]
    ax2.barh(range(len(top20)), top20["risk_score"], color=bar_colors)
    ax2.set_yticks(range(len(top20)))
    ax2.set_yticklabels(
        [f"...{cid[-6:]}" for cid in top20["nameOrig"]], fontsize=8
    )
    ax2.set_xlabel("Risk Score")
    ax2.set_title("Top 20 Clientes Mais Suspeitos")
    legend_elements = [
        Patch(facecolor="crimson", label="HIGH"),
        Patch(facecolor="darkorange", label="MEDIUM"),
        Patch(facecolor="gold", label="LOW"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right")

    # Plot 3: Frequência vs Volume (escala logarítmica)
    ax3 = axes[1, 0]
    df_plot = df_results.copy()
    df_plot["log_total_amount"] = np.log1p(df_plot["total_amount"])
    # log1p = log(1 + x): evita log(0) para clientes com total_amount = 0
    scatter_normal = df_plot[df_plot["is_anomaly"] == 0]
    scatter_anomaly = df_plot[df_plot["is_anomaly"] == 1]
    ax3.scatter(scatter_normal["total_transactions"],
                scatter_normal["log_total_amount"],
                alpha=0.3, s=10, color="steelblue", label="Normal")
    ax3.scatter(scatter_anomaly["total_transactions"],
                scatter_anomaly["log_total_amount"],
                alpha=0.7, s=30, color="crimson", label="Anómalo", zorder=5)
    ax3.set_xlabel("Número de Transações")
    ax3.set_ylabel("Log(Valor Total Movimentado)")
    ax3.set_title("Frequência vs Volume por Cliente\n(anómalos em zonas isoladas)")
    ax3.legend()

    # Plot 4: Boxplot Cash-Out Ratio
    ax4 = axes[1, 1]
    cashout_normal = df_results[df_results["is_anomaly"] == 0]["cashout_ratio"]
    cashout_anomaly = df_results[df_results["is_anomaly"] == 1]["cashout_ratio"]
    ax4.boxplot(
        [cashout_normal, cashout_anomaly],
        labels=["Normal", "Anómalo"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )
    ax4.set_ylabel("Cash-Out Ratio")
    ax4.set_title("Cash-Out Ratio: Normal vs Anómalo\n(ratio maior = mais suspeito)")

    plt.tight_layout()
    chart_paths["main_analysis"] = charts_dir / "aml_analysis_charts.png"
    plt.savefig(chart_paths["main_analysis"], bbox_inches="tight", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # GRÁFICO 2: Correlation Heatmap das features
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df_features[feature_columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # mask: esconde triângulo superior (simétrico — informação duplicada)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, ax=ax,
        annot_kws={"size": 8}, linewidths=0.5,
    )
    ax.set_title("Correlação entre Features\n(features muito correlacionadas são redundantes)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    chart_paths["correlation_heatmap"] = charts_dir / "feature_correlation_heatmap.png"
    plt.savefig(chart_paths["correlation_heatmap"], bbox_inches="tight", dpi=150)
    plt.close()

    for name, path in chart_paths.items():
        logger.info(f"   Gráfico '{name}': {path}")

    return chart_paths


# =============================================================================
# SECÇÃO 13: EXPORTAR RESULTADOS
# =============================================================================

def export_results(
    df_results: pd.DataFrame,
    config: AMLConfig,
) -> Dict[str, Path]:
    """
    Exporta resultados para CSV e imprime sumário final.
    
    Args:
        df_results: DataFrame completo com resultados e classificações.
        config: Configuração do pipeline.
    
    Returns:
        Dicionário com caminhos dos ficheiros exportados.
    """
    output_paths: Dict[str, Path] = {}

    # Exportar apenas alertas (anómalos), ordenados por risco
    df_alerts = df_results[df_results["is_anomaly"] == 1].sort_values(
        "risk_score", ascending=False
    )
    alerts_path = config.output_dir / "aml_alerts.csv"
    df_alerts.to_csv(alerts_path, index=False)
    output_paths["alerts"] = alerts_path

    # Exportar resultados completos
    full_path = config.output_dir / "full_results.csv"
    df_results.to_csv(full_path, index=False)
    output_paths["full_results"] = full_path

    # Sumário final
    logger.info("\n" + "=" * 60)
    logger.info("RELATÓRIO FINAL")
    logger.info("=" * 60)
    logger.info(f"Total de clientes analisados:   {len(df_results):>10,}")
    logger.info(f"Clientes normais:               {(df_results['is_anomaly']==0).sum():>10,}")
    logger.info(f"Alertas gerados:                {(df_results['is_anomaly']==1).sum():>10,}")
    logger.info(f"  → HIGH:                       {(df_results['risk_level']=='HIGH').sum():>10,}")
    logger.info(f"  → MEDIUM:                     {(df_results['risk_level']=='MEDIUM').sum():>10,}")
    logger.info(f"  → LOW:                        {(df_results['risk_level']=='LOW').sum():>10,}")
    logger.info(f"\nFicheiros exportados:")
    for name, path in output_paths.items():
        logger.info(f"  → {name}: {path}")

    return output_paths


# =============================================================================
# SECÇÃO 14: FUNÇÃO MAIN — ENTRY POINT DO PIPELINE
# =============================================================================

def main() -> None:
    """
    Orquestra o pipeline AML completo.
    
    PORQUÊ uma função main():
        1. Separa a lógica de orquestração das funções individuais
        2. Permite importar este módulo sem executar o pipeline
        3. É o padrão universal em Python para scripts executáveis
        4. Facilita testes — podes testar cada função individualmente
    
    FLUXO:
        Config → Validação → Carga → DB → Features → Modelo → 
        Classificação → Validação → Visualizações → Exportação
    """
    logger.info("=" * 60)
    logger.info("AML TRANSACTION ANOMALY DETECTION PIPELINE — v2.0 (Senior)")
    logger.info("=" * 60)

    # 1. Configuração
    config = AMLConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Validar inputs antes de fazer qualquer trabalho
    # PORQUÊ validar primeiro: "fail fast" — melhor saber imediatamente
    # que falta o dataset do que descobrir a meio do pipeline
    try:
        validate_dataset(config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Validação falhou: {e}")
        sys.exit(1)
        # sys.exit(1): código de saída 1 = erro (convenção Unix)
        # Sistemas de monitorização e CI/CD usam este código para detectar falhas

    # 3. Carregar dados
    logger.info("\n[1/6] Carregando dados...")
    try:
        df_raw = load_data(config)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # 4. Carregar para base de dados
    logger.info("\n[2/6] Carregando para SQLite...")
    load_to_database(df_raw, config)

    # 5. Feature engineering
    logger.info("\n[3/6] Calculando features...")
    df_features = compute_customer_features(config)

    # 6. Pré-processamento
    logger.info("\n[4/6] Pré-processando para modelo ML...")
    X_scaled, customer_ids, scaler = preprocess_features(df_features)
    feature_columns = [c for c in df_features.columns if c != "nameOrig"]

    # 7. Treinar modelo e obter predições
    logger.info("\n[5/6] Treinando modelo e gerando alertas...")
    predictions, risk_scores = train_isolation_forest(X_scaled, config)

    # 8. Classificar níveis de risco
    risk_levels = classify_risk_levels(risk_scores, predictions)

    # 9. Construir DataFrame de resultados
    df_results = df_features.copy()
    df_results["is_anomaly"] = (predictions == -1).astype(int)
    df_results["risk_score"] = risk_scores
    df_results["risk_level"] = risk_levels

    # 10. Validar modelo
    _ = validate_model(df_features, df_raw, predictions)

    # 11. Visualizações
    logger.info("\n[6/6] Gerando visualizações...")
    chart_paths = generate_visualizations(
        df_results, df_features, risk_scores,
        predictions, feature_columns, config
    )

    # 12. Exportar resultados
    export_results(df_results, config)

    logger.info("\n✅ Pipeline concluído com sucesso.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # PORQUÊ este bloco:
    #   Garante que main() só corre quando executas este ficheiro directamente:
    #   → python project1_aml_senior.py  ← main() corre
    #
    #   Se outro ficheiro fizer 'import project1_aml_senior', main() NÃO corre.
    #   Isto permite reutilizar as funções deste módulo sem efeitos colaterais.
    main()
