"""
SmartPlant AI - Asset Dataset Adapter
-------------------------------------
Interfaces with the industrial asset-level electrical energy dataset from
a manufacturing facility (Munster Technological University).

Capabilities:
1. Loads asset telemetry directly from local archive or folder.
2. Ingests sub-metered industrial machines (Presses, Compressed Air, Mixing, HVAC, Grid Incomer).
3. Adapts the 15-minute schema into SmartPlant AI's data structures.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd

DEFAULT_ZIP_PATH = r"C:\Users\sriva\Downloads\Industrial Asset-Level Electrical Energy Dataset from a Manufacturing Facility (15-Minute Aggregates).zip"
DEFAULT_FOLDER_PATH = r"C:\Users\sriva\Downloads\Industrial Asset-Level Electrical Energy Dataset from a Manufacturing Facility (15-Minute Aggregates)\Industrial Asset-Level Electrical Energy Dataset from a Manufacturing Facility (15-Minute Aggregates)"


class IndustrialAssetDatasetAdapter:
    """Adapter for reading asset-level industrial telemetry."""

    def __init__(self, zip_path=None, folder_path=None):
        self.zip_path = zip_path or DEFAULT_ZIP_PATH
        self.folder_path = folder_path or DEFAULT_FOLDER_PATH
        self.has_zip = os.path.exists(self.zip_path)
        self.has_folder = os.path.exists(self.folder_path)

        self.asset_list = None
        self.edge_list = None
        self._load_metadata()

    def _load_metadata(self):
        """Load AssetList and EdgeList metadata."""
        meta_dir = os.path.join(self.folder_path, "metadata")
        if os.path.exists(meta_dir):
            asset_path = os.path.join(meta_dir, "AssetList.csv")
            edge_path = os.path.join(meta_dir, "EdgeList.csv")
            if os.path.exists(asset_path):
                self.asset_list = pd.read_csv(asset_path)
            if os.path.exists(edge_path):
                self.edge_list = pd.read_csv(edge_path)
        elif self.has_zip:
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as z:
                    for n in z.namelist():
                        if n.endswith("AssetList.csv"):
                            with z.open(n) as fp:
                                self.asset_list = pd.read_csv(fp)
                        elif n.endswith("EdgeList.csv"):
                            with z.open(n) as fp:
                                self.edge_list = pd.read_csv(fp)
            except Exception:
                pass

    def list_assets(self):
        """Return table of all available assets with descriptions and types."""
        if self.asset_list is not None:
            return self.asset_list[['Asset_ID', 'Name', 'Description', 'Asset_Category']].copy()
        return pd.DataFrame(columns=['Asset_ID', 'Name', 'Description', 'Asset_Category'])

    def load_asset_time_series(self, asset_id="mi_a"):
        """
        Load continuous 15-minute time series for a specified asset ID.
        Asset 'mi_a' is the main electrical grid incoming feed.
        """
        if self.has_folder:
            candidate = os.path.join(self.folder_path, f"{asset_id}.csv")
            if os.path.exists(candidate):
                df = pd.read_csv(candidate)
                return self._standardize_schema(df, asset_id)

        if self.has_zip:
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as z:
                    for n in z.namelist():
                        if n.endswith(f"{asset_id}.csv") or n.endswith(f"/{asset_id}.csv"):
                            with z.open(n) as fp:
                                df = pd.read_csv(fp)
                                return self._standardize_schema(df, asset_id)
            except Exception:
                pass

        return None

    def _standardize_schema(self, df, asset_id):
        """Convert asset CSV format to standardized 15-minute telemetry."""
        df = df.copy()
        time_col = None
        for col in ['Timestamp', 'timestamp', 'Datetime', 'datetime', 'date']:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df['datetime'] = pd.to_datetime(df[time_col])
            df = df.sort_values('datetime').reset_index(drop=True)

        energy_col = None
        for col in ['Energy_kWh_15m', 'Energy_kWh', 'Usage_kWh', 'Active_Energy_kWh']:
            if col in df.columns:
                energy_col = col
                break

        if energy_col:
            df['Usage_kWh'] = pd.to_numeric(df[energy_col], errors='coerce').fillna(0.0)

        if 'datetime' in df.columns:
            df['NSM'] = df['datetime'].dt.hour * 3600 + df['datetime'].dt.minute * 60 + df['datetime'].dt.second
            df['Day_of_week'] = df['datetime'].dt.day_name()
            df['WeekStatus'] = np.where(df['datetime'].dt.dayofweek >= 5, 'Weekend', 'Weekday')

        df['asset_id'] = asset_id
        return df
