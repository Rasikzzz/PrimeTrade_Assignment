import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIGURATION & LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")

SENTIMENT_MAP: Dict[str, int] = {
    "Extreme Fear": 1,
    "Fear": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}


class CryptoSentimentAnalyzer:
    """A pipeline to merge, analyze, and visualize trading performance

    against market sentiment data.
    """

    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None

    def load_and_preprocess(
        self, trades_file: str = "historical_data.csv", sentiment_file: str = "fear_greed.csv"
    ) -> pd.DataFrame:
        """Loads trading and sentiment data, cleans timestamps, and merges them."""
        logger.info("Loading datasets...")
        try:
            trades = pd.read_csv(self.data_dir / trades_file)
            sentiment = pd.read_csv(self.data_dir / sentiment_file)
        except FileNotFoundError as e:
            logger.error(f"Data loading failed: {e}")
            raise

        # Clean and extract dates for merging
        trades["time"] = pd.to_datetime(trades["time"])
        trades["date"] = trades["time"].dt.date

        sentiment["Date"] = pd.to_datetime(sentiment["Date"])
        sentiment["date"] = sentiment["Date"].dt.date

        logger.info("Merging datasets on date...")
        self.df = trades.merge(
            sentiment[["date", "Classification"]], on="date", how="left"
        )
        return self.df

    def feature_engineering(self) -> pd.DataFrame:
        """Generates win flags and maps categorical sentiment to numerical scores."""
        if self.df is None or self.df.empty:
            raise ValueError("No data available. Run load_and_preprocess() first.")

        logger.info("Engineering features...")
        self.df["win"] = self.df["closedPnL"] > 0
        self.df["sentiment_score"] = self.df["Classification"].map(SENTIMENT_MAP)

        missing_sentiment = self.df["sentiment_score"].isna().sum()
        if missing_sentiment > 0:
            logger.warning(f"Found {missing_sentiment} rows with unmapped sentiment classifications.")

        return self.df

    def run_analysis(self) -> None:
        """Computes and logs key trading metrics aggregated by market sentiment."""
        if self.df is None:
            raise ValueError("DataFrame is not initialized.")

        logger.info("Computing aggregation metrics...")

        # Consolidated group-by matrix for efficiency
        metrics_by_sentiment = self.df.groupby("Classification").agg(
            total_trades=("closedPnL", "count"),
            avg_pnl=("closedPnL", "mean"),
            total_pnl=("closedPnL", "sum"),
            win_rate=("win", lambda x: np.mean(x) * 100),
            avg_leverage=("leverage", "mean"),
        )

        print("\n" + "=" * 50)
        print("          TRADING METRICS BY SENTIMENT          ")
        print("=" * 50)
        print(metrics_by_sentiment.to_string())

        print("\n" + "=" * 50)
        print("          LONG / SHORT DISTRIBUTION             ")
        print("=" * 50)
        print(pd.crosstab(self.df["Classification"], self.df["side"]))

        print("\n" + "=" * 50)
        print("          TOP 10 TRADERS BY PnL                 ")
        print("=" * 50)
        top_traders = (
            self.df.groupby("account")["closedPnL"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        print(top_traders.to_string())
        print("=" * 50 + "\n")

    def generate_plots(self) -> None:
        """Generates and saves analytical plots to the output directory."""
        if self.df is None:
            raise ValueError("DataFrame is not initialized.")

        logger.info("Generating and saving visualizations...")

        # Plot 1: PnL Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self.df, x="Classification", y="closedPnL", ax=ax)
        ax.set_title("PnL Distribution by Sentiment", fontsize=14, pad=15)
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(self.output_dir / "pnl_sentiment.png", dpi=300)
        plt.close(fig)

        # Plot 2: Leverage
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df, x="Classification", y="leverage", ax=ax)
        ax.set_title("Average Leverage by Sentiment", fontsize=14, pad=15)
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(self.output_dir / "leverage_sentiment.png", dpi=300)
        plt.close(fig)

        # Plot 3: Correlation Matrix
        corr_cols = ["sentiment_score", "closedPnL", "leverage"]
        corr = self.df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14, pad=15)
        fig.tight_layout()
        fig.savefig(self.output_dir / "correlation.png", dpi=300)
        plt.close(fig)

    def export_data(self, filename: str = "merged_analysis.csv") -> None:
        """Exports the processed dataframe to a CSV file."""
        if self.df is None:
            raise ValueError("DataFrame is not initialized.")

        output_path = self.output_dir / filename
        logger.info(f"Exporting processed data to {output_path}...")
        self.df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# EXECUTION ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the pipeline
    analyzer = CryptoSentimentAnalyzer(data_dir="data", output_dir="output")

    try:
        analyzer.load_and_preprocess()
        analyzer.feature_engineering()
        analyzer.run_analysis()
        analyzer.generate_plots()
        analyzer.export_data()
        logger.info("Analysis pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Pipeline failed due to an unhandled error: {e}", exc_info=True)
