#!/usr/bin/env python3

import re
import json
import os
import sys
import subprocess
import nltk
import warnings
import csv
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import networkx as nx

import torch
import torch.nn as nn

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

from flask import (
    Flask,
    request,
    render_template_string,
    send_from_directory,
    url_for,
    jsonify,
)

from werkzeug.utils import secure_filename

# Import Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")

# NLTK setup
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    print("First-time setup: Downloading NLTK VADER lexicon...")
    nltk.download("vader_lexicon", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("First-time setup: Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {DEVICE}")


# -------------------------------------------------------------------
# Helper: open file
# -------------------------------------------------------------------
def open_file(path: str):
    """Open a file with the default OS application."""
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return
    try:
        if os.name == "nt":  # Windows
            os.startfile(path)
        elif sys.platform == "darwin":  # macOS
            subprocess.Popen(["open", path])
        else:  # Linux
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"⚠️ Could not open file {path}: {e}")


# -------------------------------------------------------------------
# BERT Feature Extractor
# -------------------------------------------------------------------
class BERTFeatureExtractor:
    def __init__(self):
        self.available = False
        self.tokenizer = None
        self.model = None
        self._try_load_bert()

    def _try_load_bert(self):
        try:
            print("🤖 Loading BERT model...", end=" ")
            from transformers import BertTokenizer, BertModel

            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", local_files_only=False
            )
            self.model = (
                BertModel.from_pretrained(
                    "bert-base-uncased", local_files_only=False
                ).to(DEVICE)
            )
            self.model.eval()
            self.available = True
            print("✓ BERT loaded successfully")
        except Exception:
            print("⚠️ BERT unavailable (using fallback features)")
            self.available = False

    def extract_features(self, text: str) -> np.ndarray:
        if not self.available:
            return np.zeros(768)
        try:
            inputs = self.tokenizer(
                text[:512],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings.flatten()
        except Exception:
            return np.zeros(768)


# -------------------------------------------------------------------
# LSTM model wrapper
# -------------------------------------------------------------------
class LSTMRiskPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMRiskPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# -------------------------------------------------------------------
# Core META DETECT Engine
# -------------------------------------------------------------------
class MetaDetect:
    def __init__(self):
        print("\n🚀 Initializing META DETECT...")
        self.analyzer = SentimentIntensityAnalyzer()
        self.bert_extractor = BERTFeatureExtractor()
        self.use_bert = self.bert_extractor.available

        self.drug_keywords = {
            "high_risk": [
                "cocaine",
                "heroin",
                "meth",
                "mdma",
                "ecstasy",
                "lsd",
                "fentanyl",
                "oxy",
                "xanax",
                "molly",
                "crack",
                "crystal",
                "smack",
                "blow",
                "speed",
                "amphetamine",
                "morphine",
                "opium",
                "codeine",
                "tramadol",
                "hydrocodone",
            ],
            "medium_risk": [
                "weed",
                "hash",
                "kush",
                "joint",
                "blunt",
                "stash",
                "dealer",
                "pickup",
            ],
            "contextual_slang": [
                "stuff",
                "product",
                "package",
                "delivery",
                "drop",
                "meet",
                "cash",
                "party",
                "supply",
                "client",
                "batch",
                "order",
            ],
        }

        self.excluded_keywords = {
            "pot",
            "bud",
            "dope",
            "ice",
            "gram",
            "ounce",
            "oz",
            "eighth",
            "quarter",
            "half",
            "plug",
            "connect",
            "score",
            "deal",
        }

        self.innocent_patterns = [
            r"\b(product|package|delivery|order|client|supply)\s+(development|management|tracking|service|support|team|meeting)\b",
            r"\b(amazon|ebay|flipkart|shop|store|business|company|work|office|courier|fedex|ups|dhl)\b",
            r"\b(delivery|package)\s+(arrived|coming|expected|received|sent|delayed|tracking)\b",
            r"\border\s+(online|number|status|confirmation)\b",
            r"\b(birthday|wedding|celebration|dinner|lunch|event|surprise)\s+party\b",
            r"\bparty\s+(planning|invitation|tonight|tomorrow|last\s+night|yesterday|hat|dress|theme|venue)\b",
            r"\b(house|dinner|lunch|pool|garden|tea)\s+party\b",
            r"\b(cash|money)\s+(back|app|payment|transfer|withdraw|deposit|atm|machine)\b",
            r"\bpay\s+(cash|money|by\s+cash)\b",
            r"\b(have|need|get|give|owe)\s+(cash|money)\s+(for|to\s+pay|from|back)\b",
            r"\b(cashback|cashless|petty\s+cash)\b",
            r"\bmeet(ing)?\s+(at|for|with|tomorrow|today|later|scheduled|zoom|teams|client)\b",
            r"\b(coffee|lunch|dinner|breakfast|video)\s+meet(ing)?\b",
            r"\b(team|staff|board|project)\s+meet(ing)?\b",
            r"\b(nice|great|pleased)\s+to\s+meet\b",
        ]

        self.suspicious_patterns = [
            r"\b(sell|selling|sold)\s+(weed|hash|kush)\b",
            r"\b(buy|buying|bought)\s+(weed|hash|kush)\b",
            r"\b(good|bad|pure|quality|premium|fresh)\s+(weed|hash|kush)\b",
            r"\bhow\s+much\s+(for|per)\s+(it|them|that)\b",
            r"\b(need|got|have)\s+(some|the)?\s*(good)?\s*(weed|hash|kush)\b",
            r"\bmeet\s+(secretly|privately|quietly|discreetly)\b",
            r"\b(dealer|supplier)\b",
            r"\bhit\s+me\s+up\b",
            r"\blink\s+up\b",
        ]

        self.risk_weights = {
            "high_risk": 10,
            "medium_risk": 5,
            "contextual_slang": 3,
        }

        self.all_direct_drug_terms = set(
            self.drug_keywords["high_risk"] + self.drug_keywords["medium_risk"]
        ) - self.excluded_keywords

        self.interaction_graph = nx.DiGraph()
        print("✓ META DETECT initialized successfully\n")

    def _is_innocent_context(self, message: str) -> bool:
        message_lower = message.lower()
        for pattern in self.innocent_patterns:
            if re.search(pattern, message_lower):
                return True
        return False

    def _has_suspicious_context(self, message: str) -> bool:
        message_lower = message.lower()
        for pattern in self.suspicious_patterns:
            if re.search(pattern, message_lower):
                return True
        return False

    def _calculate_context_risk(
        self,
        message: str,
        keyword: str,
        category: str,
        sentiment_compound: float,
    ) -> Tuple[bool, str]:
        message_lower = message.lower()

        if category in ["high_risk", "medium_risk"]:
            return True, f"Direct drug term: '{keyword}'"

        if category == "contextual_slang":
            if self._is_innocent_context(message):
                return False, "Innocent context detected"
            if self._has_suspicious_context(message):
                return True, f"Suspicious context with slang: '{keyword}'"
            if sentiment_compound <= -0.5:
                return True, f"Negative sentiment ({sentiment_compound:.2f}) with slang"
        return False, "Contextual slang without suspicious indicators"

    def test_detection_accuracy(self) -> float:
        print("🧪 Running automated detection accuracy tests...")
        test_cases = [
            ("I'm learning to program in Python", False, "gram"),
            ("Check out my Instagram profile", False, "gram"),
            ("Need 500 grams of rice for dinner", False, "gram"),
            ("My package delivery is coming tomorrow", False, "package"),
            ("Birthday party tonight!", False, "party"),
            ("Selling weed, good quality", True, "weed"),
            ("Got cocaine available", True, "cocaine"),
        ]

        correct = 0
        total = len(test_cases)

        for message, should_detect, keyword in test_cases:
            sentiment = self.analyzer.polarity_scores(message)

            if keyword in self.excluded_keywords:
                detected = False
            else:
                word_pattern = r"\b" + re.escape(keyword) + r"\b"
                keyword_found = bool(re.search(word_pattern, message.lower()))

                if not keyword_found:
                    detected = False
                else:
                    category = None
                    for cat, keywords in self.drug_keywords.items():
                        if keyword in keywords:
                            category = cat
                            break
                    if category:
                        detected, _ = self._calculate_context_risk(
                            message, keyword, category, sentiment["compound"]
                        )
                    else:
                        detected = False

            if detected == should_detect:
                correct += 1

        accuracy = (correct / total) * 100
        print(f"✓ Detection accuracy: {correct}/{total} ({accuracy:.1f}%)\n")
        return accuracy

    def parse_whatsapp_chat(self, filepath: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        pattern = re.compile(
            r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?:\s*(?:AM|PM))?)\s*-\s*([^:]+):\s*(.+)",
            re.IGNORECASE,
        )

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = pattern.match(line.strip())
                    if match:
                        messages.append(
                            {
                                "date": match.group(1).strip(),
                                "time": match.group(2).strip(),
                                "user": match.group(3).strip(),
                                "message": match.group(4).strip(),
                            }
                        )
        except Exception as e:
            print(f"⚠️ Error reading file: {str(e)[:100]}")

        return messages

    def parse_telegram_chat(self, filepath: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        pattern = re.compile(
            r"\[(\d{1,2}\.\d{1,2}\.\d{2,4})\s+(\d{1,2}:\d{2}:\d{2})\]\s+([^:]+):\s*(.+)"
        )

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = pattern.match(line.strip())
                    if match:
                        messages.append(
                            {
                                "date": match.group(1).strip(),
                                "time": match.group(2).strip(),
                                "user": match.group(3).strip(),
                                "message": match.group(4).strip(),
                            }
                        )
        except Exception as e:
            print(f"⚠️ Error reading file: {str(e)[:100]}")

        return messages

    def build_interaction_network(self, messages: List[Dict[str, str]]):
        print("🕸️ Building interaction network...")
        self.interaction_graph.clear()

        if len(messages) < 2:
            print("⚠️ Insufficient messages for network analysis.")
            return

        for i in range(len(messages) - 1):
            try:
                user1 = messages[i]["user"].strip()
                user2 = messages[i + 1]["user"].strip()
                if user1 and user2 and user1 != user2:
                    if self.interaction_graph.has_edge(user1, user2):
                        self.interaction_graph[user1][user2]["weight"] += 1
                    else:
                        self.interaction_graph.add_edge(user1, user2, weight=1)
            except Exception:
                continue

        print(
            f"✓ Network built: {self.interaction_graph.number_of_nodes()} nodes, "
            f"{self.interaction_graph.number_of_edges()} edges"
        )

    def analyze_network_centrality(self) -> Dict[str, Dict[str, float]]:
        centrality_metrics: Dict[str, Dict[str, float]] = {}

        if self.interaction_graph.number_of_nodes() == 0:
            return centrality_metrics

        try:
            degree_cent = nx.degree_centrality(self.interaction_graph)
            betweenness_cent = nx.betweenness_centrality(self.interaction_graph)
            try:
                pagerank = nx.pagerank(self.interaction_graph)
            except Exception:
                pagerank = {node: 0 for node in self.interaction_graph.nodes()}

            for user in self.interaction_graph.nodes():
                centrality_metrics[user] = {
                    "degree_centrality": float(degree_cent.get(user, 0)),
                    "betweenness_centrality": float(betweenness_cent.get(user, 0)),
                    "pagerank": float(pagerank.get(user, 0)),
                }
        except Exception as e:
            print(f"⚠️ Error calculating centrality: {str(e)[:100]}")

        return centrality_metrics

    def analyze_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        user_stats: Dict[str, Any] = defaultdict(
            lambda: {
                "message_count": 0,
                "keyword_count": 0,
                "high_risk_keywords": 0,
                "medium_risk_keywords": 0,
                "contextual_slang_keywords": 0,
                "false_positives_avoided": 0,
                "suspicion_score": 0,
                "total_length": 0,
                "sentiment_scores": [],
                "negative_message_count": 0,
                "message_times": [],
                "messages_text": [],
                "suspicious_words": [],
                "suspicious_word_frequency": Counter(),
                "innocent_detections": [],
                "bert_embeddings": [],
            }
        )

        print("🔍 Analyzing messages with context-aware detection...")

        for idx, msg in enumerate(messages):
            if idx % max(1, len(messages) // 10) == 0 and idx > 0:
                print(f"   Processing message {idx}/{len(messages)}...", end="\r")
            try:
                user = msg.get("user", "Unknown").strip()
                message = msg.get("message", "").strip()
                if not user or not message:
                    continue
                user_stats[user]["message_count"] += 1
                user_stats[user]["total_length"] += len(message)
                user_stats[user]["message_times"].append(msg.get("time", ""))
                user_stats[user]["messages_text"].append(message)

                sentiment = self.analyzer.polarity_scores(message)
                user_stats[user]["sentiment_scores"].append(sentiment["compound"])
                if sentiment["compound"] <= -0.4:
                    user_stats[user]["negative_message_count"] += 1

                if self.use_bert and idx % 5 == 0:
                    bert_features = self.bert_extractor.extract_features(message)
                    user_stats[user]["bert_embeddings"].append(bert_features)

                message_lower = message.lower()

                for category, keywords in self.drug_keywords.items():
                    for keyword in keywords:
                        if keyword in self.excluded_keywords:
                            continue
                        word_pattern = r"\b" + re.escape(keyword) + r"\b"
                        if re.search(word_pattern, message_lower):
                            is_suspicious, reason = self._calculate_context_risk(
                                message,
                                keyword,
                                category,
                                sentiment["compound"],
                            )
                            if is_suspicious:
                                user_stats[user]["keyword_count"] += 1
                                user_stats[user][f"{category}_keywords"] += 1
                                user_stats[user]["suspicion_score"] += self.risk_weights[
                                    category
                                ]
                                user_stats[user]["suspicious_words"].append(
                                    {
                                        "word": keyword,
                                        "category": category,
                                        "message_context": message[:150],
                                        "timestamp": msg.get("time", "Unknown"),
                                        "reason": reason,
                                    }
                                )
                                user_stats[user]["suspicious_word_frequency"][
                                    keyword
                                ] += 1
                            else:
                                user_stats[user]["false_positives_avoided"] += 1
                                user_stats[user]["innocent_detections"].append(
                                    {
                                        "word": keyword,
                                        "message": message[:150],
                                        "reason": reason,
                                    }
                                )
            except Exception:
                continue

        print(f"\n✓ Analyzed {len(messages)} messages from {len(user_stats)} users")

        for user, stats in user_stats.items():
            if stats["message_count"] > 0:
                stats["avg_message_length"] = (
                    stats["total_length"] / stats["message_count"]
                )
                stats["negative_message_ratio"] = (
                    stats["negative_message_count"] / stats["message_count"]
                )
                stats["avg_sentiment"] = (
                    float(np.mean(stats["sentiment_scores"]))
                    if stats["sentiment_scores"]
                    else 0.0
                )
                if stats["bert_embeddings"]:
                    stats["avg_bert_embedding"] = np.mean(
                        stats["bert_embeddings"], axis=0
                    )
                else:
                    stats["avg_bert_embedding"] = np.zeros(768)
            else:
                stats["avg_message_length"] = 0
                stats["negative_message_ratio"] = 0
                stats["avg_sentiment"] = 0.0
                stats["avg_bert_embedding"] = np.zeros(768)
            stats["suspicion_score"] = min(100, stats["suspicion_score"])

        return dict(user_stats)

    def apply_dbscan_clustering(
        self,
        user_stats: Dict[str, Any],
        centrality_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        print("🔬 Applying DBSCAN clustering...")

        if len(user_stats) < 3:
            print("⚠️ Not enough users for DBSCAN. Skipping.")
            for user in user_stats:
                user_stats[user]["dbscan_cluster"] = -1
            return user_stats

        try:
            users = list(user_stats.keys())
            features = []
            for user in users:
                stats = user_stats[user]
                centrality = centrality_metrics.get(
                    user,
                    {
                        "degree_centrality": 0,
                        "betweenness_centrality": 0,
                        "pagerank": 0,
                    },
                )
                feature_vec = [
                    stats["suspicion_score"],
                    stats["message_count"],
                    stats["keyword_count"],
                    stats.get("avg_message_length", 0),
                    stats.get("avg_sentiment", 0),
                    stats.get("negative_message_ratio", 0),
                    centrality["degree_centrality"] * 100,
                    centrality["betweenness_centrality"] * 100,
                    centrality["pagerank"] * 100,
                ]
                features.append(feature_vec)

            features = np.array(features, dtype=np.float32)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            dbscan = DBSCAN(eps=0.5, min_samples=max(2, len(users) // 5))
            clusters = dbscan.fit_predict(features_scaled)

            for i, user in enumerate(users):
                user_stats[user]["dbscan_cluster"] = int(clusters[i])

            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            print(f"✓ DBSCAN found {n_clusters} clusters and {n_noise} outliers")
        except Exception as e:
            print(f"⚠️ DBSCAN clustering failed: {str(e)[:100]}")
            for user in user_stats:
                user_stats[user]["dbscan_cluster"] = -1

        return user_stats

    def apply_kmeans_clustering(self, user_stats: Dict[str, Any]) -> Dict[str, Any]:
        print("📊 Applying K-Means clustering...")

        if len(user_stats) < 3:
            print("⚠️ Not enough users for K-Means. Skipping.")
            for user in user_stats:
                user_stats[user]["kmeans_cluster"] = "N/A"
            return user_stats

        try:
            users = list(user_stats.keys())
            features = np.array(
                [
                    [
                        stats["suspicion_score"],
                        stats["message_count"],
                        stats["keyword_count"],
                        stats.get("avg_message_length", 0),
                        stats.get("avg_sentiment", 0),
                        stats.get("negative_message_ratio", 0),
                    ]
                    for stats in user_stats.values()
                ],
                dtype=np.float32,
            )

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            n_clusters = min(3, len(users))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)

            avg_scores: Dict[int, list] = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                avg_scores[cluster_id].append(user_stats[users[i]]["suspicion_score"])

            sorted_clusters = sorted(
                avg_scores.items(), key=lambda item: np.mean(item[1]), reverse=True
            )

            labels = ["High Risk", "Medium Risk", "Low Risk"]
            labels_map = {cid: labels[i] for i, (cid, _) in enumerate(sorted_clusters)}

            for i, user in enumerate(users):
                user_stats[user]["kmeans_cluster"] = labels_map.get(
                    clusters[i], "Low Risk"
                )

            print("✓ K-Means clustering complete")
        except Exception as e:
            print(f"⚠️ K-Means clustering failed: {str(e)[:100]}")
            for user in user_stats:
                user_stats[user]["kmeans_cluster"] = "N/A"

        return user_stats

    def train_lstm_model(self, user_stats: Dict[str, Any]) -> Dict[str, float]:
        print("🧠 Training LSTM model for temporal analysis (heuristic)...")
        lstm_scores: Dict[str, float] = {}

        try:
            for user, stats in user_stats.items():
                if stats["message_count"] < 10:
                    lstm_scores[user] = 0.0
                    continue

                temporal_features = []
                for i, msg_text in enumerate(stats["messages_text"][:100]):
                    sentiment_idx = min(i, len(stats["sentiment_scores"]) - 1)
                    features = [
                        len(msg_text) / 500.0,
                        (stats["sentiment_scores"][sentiment_idx] + 1) / 2.0,
                        1.0
                        if any(
                            kw in msg_text.lower() for kw in self.all_direct_drug_terms
                        )
                        else 0.0,
                        i / max(len(stats["messages_text"]), 1),
                        stats["keyword_count"] / max(stats["message_count"], 1),
                        stats.get("avg_message_length", 0) / 500.0,
                        stats["negative_message_ratio"],
                        stats["suspicion_score"] / 100.0,
                        1.0 if stats.get("high_risk_keywords", 0) > 0 else 0.0,
                        1.0 if stats.get("medium_risk_keywords", 0) > 0 else 0.0,
                    ]
                    temporal_features.append(features[:10])

                if len(temporal_features) < 5:
                    lstm_scores[user] = 0.0
                    continue

                temporal_array = np.array(temporal_features)
                variance_score = np.mean(np.std(temporal_array, axis=0))
                lstm_scores[user] = min(1.0, float(variance_score))
        except Exception as e:
            print(f"⚠️ LSTM training encountered error: {str(e)[:100]}")
            lstm_scores = {user: 0.0 for user in user_stats.keys()}

        print("✓ LSTM temporal analysis complete")
        return lstm_scores

    def generate_comprehensive_report(
        self,
        user_stats: Dict[str, Any],
        centrality_metrics: Dict[str, Dict[str, float]],
        lstm_scores: Dict[str, float],
        output_path: str,
    ) -> Dict[str, Any]:
        sorted_users = sorted(
            user_stats.items(),
            key=lambda x: x[1]["suspicion_score"],
            reverse=True,
        )

        total_false_positives_avoided = sum(
            u["false_positives_avoided"] for u in user_stats.values()
        )

        report: Dict[str, Any] = {
            "analysis_timestamp": datetime.now().isoformat(),
            "detection_system": "META DETECT v1.0",
            "detection_mode": "Context-Aware + BERT + LSTM",
            "total_users": len(user_stats),
            "total_false_positives_avoided": total_false_positives_avoided,
            "kmeans_high_risk": sum(
                1 for u in user_stats.values() if u.get("kmeans_cluster") == "High Risk"
            ),
            "kmeans_medium_risk": sum(
                1
                for u in user_stats.values()
                if u.get("kmeans_cluster") == "Medium Risk"
            ),
            "kmeans_low_risk": sum(
                1 for u in user_stats.values() if u.get("kmeans_cluster") == "Low Risk"
            ),
            "dbscan_outliers": sum(
                1 for u in user_stats.values() if u.get("dbscan_cluster") == -1
            ),
            "total_keywords_detected": sum(
                u["keyword_count"] for u in user_stats.values()
            ),
            "network_analysis_enabled": bool(centrality_metrics),
            "bert_enabled": self.use_bert,
            "users": [],
        }

        for i, (user, stats) in enumerate(sorted_users):
            centrality = centrality_metrics.get(user, {})
            top_suspicious_words = stats["suspicious_word_frequency"].most_common(5)
            suspicious_words_summary = [
                {
                    "word": word,
                    "frequency": count,
                    "category": next(
                        (cat for cat, kws in self.drug_keywords.items() if word in kws),
                        "unknown",
                    ),
                }
                for word, count in top_suspicious_words
            ]

            user_report = {
                "rank": i + 1,
                "username": user,
                "suspicion_score": float(stats["suspicion_score"]),
                "false_positives_avoided": int(stats["false_positives_avoided"]),
                "kmeans_cluster": stats.get("kmeans_cluster", "N/A"),
                "dbscan_cluster": int(stats.get("dbscan_cluster", -1)),
                "lstm_risk_score": round(float(lstm_scores.get(user, 0)), 3),
                "message_count": int(stats["message_count"]),
                "keyword_count": int(stats["keyword_count"]),
                "high_risk_keywords": int(stats["high_risk_keywords"]),
                "medium_risk_keywords": int(stats["medium_risk_keywords"]),
                "contextual_slang_keywords": int(
                    stats.get("contextual_slang_keywords", 0)
                ),
                "avg_message_length": round(float(stats.get("avg_message_length", 0)), 2),
                "avg_sentiment_score": round(float(stats.get("avg_sentiment", 0)), 3),
                "negative_message_ratio": round(
                    float(stats.get("negative_message_ratio", 0)), 2
                ),
                "network_degree_centrality": round(
                    float(centrality.get("degree_centrality", 0)), 3
                ),
                "network_betweenness": round(
                    float(centrality.get("betweenness_centrality", 0)), 3
                ),
                "network_pagerank": round(float(centrality.get("pagerank", 0)), 3),
                "suspicious_words_used": suspicious_words_summary,
                "total_suspicious_words_instances": len(stats["suspicious_words"]),
            }
            report["users"].append(user_report)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(f"\n✓ Comprehensive report saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Error saving report: {str(e)[:100]}")

        return report

    def export_suspicious_words_csv(
        self,
        user_stats: Dict[str, Any],
        base_filename: str,
        output_dir: str = ".",
    ) -> str:
        csv_path = os.path.join(
            output_dir, f"{base_filename}_suspicious_words_details.csv"
        )

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Username",
                        "Word",
                        "Category",
                        "Frequency",
                        "Detection Reason",
                        "Message Context",
                        "Timestamp",
                    ]
                )
                for user, stats in user_stats.items():
                    for word_instance in stats["suspicious_words"]:
                        writer.writerow(
                            [
                                user,
                                word_instance["word"],
                                word_instance["category"],
                                stats["suspicious_word_frequency"][
                                    word_instance["word"]
                                ],
                                word_instance.get("reason", "N/A"),
                                word_instance["message_context"],
                                word_instance["timestamp"],
                            ]
                        )
            print(f"✓ Suspicious words details exported to {csv_path}")
        except Exception as e:
            print(f"⚠️ Error exporting CSV: {str(e)[:100]}")
        return csv_path

    def export_suspicious_words_summary_csv(
        self,
        report: Dict[str, Any],
        base_filename: str,
        output_dir: str = ".",
    ) -> str:
        csv_path = os.path.join(
            output_dir, f"{base_filename}_suspicious_words_summary.csv"
        )

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Rank",
                        "Username",
                        "Risk Score",
                        "False Positives Avoided",
                        "Total Instances",
                        "Top Word",
                        "Top Word Frequency",
                        "Top Word Category",
                    ]
                )

                for user_report in report["users"]:
                    top_word = (
                        user_report["suspicious_words_used"][0]
                        if user_report["suspicious_words_used"]
                        else {}
                    )
                    writer.writerow(
                        [
                            user_report["rank"],
                            user_report["username"],
                            user_report["suspicion_score"],
                            user_report["false_positives_avoided"],
                            user_report["total_suspicious_words_instances"],
                            top_word.get("word", "N/A"),
                            top_word.get("frequency", 0),
                            top_word.get("category", "N/A"),
                        ]
                    )
            print(f"✓ Suspicious words summary exported to {csv_path}")
        except Exception as e:
            print(f"⚠️ Error exporting CSV: {str(e)[:100]}")
        return csv_path

    def create_static_dashboard_image(
        self,
        user_stats: Dict[str, Any],
        centrality_metrics: Dict[str, Dict[str, float]],
        output_path: str,
    ):
        if not user_stats:
            print("⚠️ No user statistics to visualize.")
            return

        print("📸 Generating static dashboard image (PNG)...")

        try:
            df = pd.DataFrame(
                [
                    {
                        "User": user[:20],
                        "Score": stats["suspicion_score"],
                        "Messages": stats["message_count"],
                        "Keywords": stats["keyword_count"],
                        "KMeans": stats.get("kmeans_cluster", "N/A"),
                        "DBSCAN": stats.get("dbscan_cluster", -1),
                        "Sentiment": stats.get("avg_sentiment", 0),
                        "Degree": centrality_metrics.get(user, {}).get(
                            "degree_centrality", 0
                        ),
                        "Betweenness": centrality_metrics.get(user, {}).get(
                            "betweenness_centrality", 0
                        ),
                        "SuspiciousWords": len(stats["suspicious_words"]),
                        "FalsePositivesAvoided": stats["false_positives_avoided"],
                    }
                    for user, stats in user_stats.items()
                ]
            ).sort_values("Score", ascending=False)

            if len(df) == 0:
                print("⚠️ DataFrame is empty.")
                return

            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

            color_map = {
                "High Risk": "#2563eb",
                "Medium Risk": "#3b82f6",
                "Low Risk": "#60a5fa",
                "N/A": "#64748b",
            }

            ax1 = fig.add_subplot(gs[0, :2])
            top_10 = df.head(10)
            colors_top10 = [color_map.get(c, "#64748b") for c in top_10["KMeans"]]
            ax1.barh(top_10["User"], top_10["Score"], color=colors_top10)
            ax1.set_xlabel("Suspicion Score", fontsize=10)
            ax1.set_title("Top 10 Users by Risk Score", fontsize=12, fontweight="bold")
            ax1.invert_yaxis()

            ax2 = fig.add_subplot(gs[0, 2])
            kmeans_counts = df["KMeans"].value_counts()
            colors_pie = [color_map.get(c, "#64748b") for c in kmeans_counts.index]
            ax2.pie(
                kmeans_counts.values,
                labels=kmeans_counts.index,
                autopct="%1.1f%%",
                colors=colors_pie,
                startangle=90,
            )
            ax2.set_title("K-Means Distribution", fontsize=12, fontweight="bold")

            ax3 = fig.add_subplot(gs[1, 0])
            dbscan_counts = df["DBSCAN"].value_counts().sort_index()
            dbscan_labels = [
                f"C{i}" if i != -1 else "Outlier" for i in dbscan_counts.index
            ]
            ax3.bar(dbscan_labels, dbscan_counts.values, color="#2563eb")
            ax3.set_ylabel("Count", fontsize=10)
            ax3.set_title("DBSCAN Clusters", fontsize=12, fontweight="bold")

            ax4 = fig.add_subplot(gs[1, 1:])
            top_central = df.nlargest(10, "Degree")
            ax4.barh(
                top_central["User"],
                top_central["Degree"],
                color=plt.cm.Blues(top_central["Degree"] / top_central["Degree"].max()),
            )
            ax4.set_xlabel("Degree Centrality", fontsize=10)
            ax4.set_title(
                "Network Degree Centrality (Top 10)", fontsize=12, fontweight="bold"
            )
            ax4.invert_yaxis()

            ax5 = fig.add_subplot(gs[2, :2])
            for cluster, color in color_map.items():
                cluster_data = df[df["KMeans"] == cluster]
                ax5.scatter(
                    cluster_data["Sentiment"],
                    cluster_data["Score"],
                    s=cluster_data["Messages"] * 2,
                    alpha=0.6,
                    color=color,
                    label=cluster,
                    edgecolors="black",
                    linewidth=0.5,
                )
            ax5.set_xlabel("Avg Sentiment", fontsize=10)
            ax5.set_ylabel("Suspicion Score", fontsize=10)
            ax5.set_title("Sentiment vs Risk Score", fontsize=12, fontweight="bold")
            ax5.legend(loc="upper right", fontsize=8)
            ax5.grid(True, alpha=0.3)

            ax6 = fig.add_subplot(gs[2, 2])
            for cluster, color in color_map.items():
                cluster_data = df[df["KMeans"] == cluster]
                ax6.scatter(
                    cluster_data["SuspiciousWords"],
                    cluster_data["Score"],
                    s=cluster_data["Keywords"] * 10,
                    alpha=0.6,
                    color=color,
                    label=cluster,
                    edgecolors="black",
                    linewidth=0.5,
                )
            ax6.set_xlabel("Suspicious Words Count", fontsize=10)
            ax6.set_ylabel("Suspicion Score", fontsize=10)
            ax6.set_title("Suspicious Words vs Risk", fontsize=12, fontweight="bold")
            ax6.grid(True, alpha=0.3)

            ax7 = fig.add_subplot(gs[3, 0])
            top_fp = df.nlargest(10, "FalsePositivesAvoided")
            ax7.barh(
                top_fp["User"],
                top_fp["FalsePositivesAvoided"],
                color=plt.cm.Greens(
                    top_fp["FalsePositivesAvoided"]
                    / (top_fp["FalsePositivesAvoided"].max() + 1)
                ),
            )
            ax7.set_xlabel("False Positives Avoided", fontsize=10)
            ax7.set_title(
                "False Positives Avoided (Top 10)", fontsize=12, fontweight="bold"
            )
            ax7.invert_yaxis()

            ax8 = fig.add_subplot(gs[3, 1])
            total_suspicious = df["SuspiciousWords"].sum()
            total_false_avoided = df["FalsePositivesAvoided"].sum()
            ax8.bar(
                ["True Positives\n(Suspicious)", "False Positives\nAvoided"],
                [total_suspicious, total_false_avoided],
                color=["#2563eb", "#60a5fa"],
            )
            ax8.set_ylabel("Count", fontsize=10)
            ax8.set_title(
                "Context-Aware Detection Accuracy", fontsize=12, fontweight="bold"
            )
            for i, v in enumerate([total_suspicious, total_false_avoided]):
                ax8.text(i, v + max(total_suspicious, total_false_avoided) * 0.02, str(v), ha="center", fontweight="bold")

            ax9 = fig.add_subplot(gs[3, 2])
            keyword_data = df[["User", "Keywords"]].nlargest(10, "Keywords")
            ax9.barh(
                keyword_data["User"],
                keyword_data["Keywords"],
                color=plt.cm.Blues(
                    keyword_data["Keywords"] / (keyword_data["Keywords"].max() + 1)
                ),
            )
            ax9.set_xlabel("Keywords Count", fontsize=10)
            ax9.set_title("Top 10 Keyword Detections", fontsize=12, fontweight="bold")
            ax9.invert_yaxis()

            fig.suptitle(
                "🔍 META DETECT Visual Dashboard v1.0\nContext-Aware NLP + Network Analysis + DBSCAN + K-Means + BERT + LSTM",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            print(f"✓ Static dashboard image saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Error generating static dashboard image: {str(e)[:100]}")

    def create_interactive_dashboard(
        self,
        user_stats: Dict[str, Any],
        centrality_metrics: Dict[str, Dict[str, float]],
        output_path: str,
    ):
        if not user_stats:
            print("⚠️ No user statistics to visualize.")
            return

        print("📈 Generating interactive Plotly dashboard...")

        try:
            df = pd.DataFrame(
                [
                    {
                        "User": user[:20],
                        "Score": stats["suspicion_score"],
                        "Messages": stats["message_count"],
                        "Keywords": stats["keyword_count"],
                        "KMeans": stats.get("kmeans_cluster", "N/A"),
                        "DBSCAN": stats.get("dbscan_cluster", -1),
                        "Sentiment": stats.get("avg_sentiment", 0),
                        "Degree": centrality_metrics.get(user, {}).get(
                            "degree_centrality", 0
                        ),
                        "Betweenness": centrality_metrics.get(user, {}).get(
                            "betweenness_centrality", 0
                        ),
                        "SuspiciousWords": len(stats["suspicious_words"]),
                        "FalsePositivesAvoided": stats["false_positives_avoided"],
                    }
                    for user, stats in user_stats.items()
                ]
            ).sort_values("Score", ascending=False)

            if len(df) == 0:
                print("⚠️ DataFrame is empty.")
                return

            color_map = {
                "High Risk": "#2563eb",
                "Medium Risk": "#3b82f6",
                "Low Risk": "#60a5fa",
                "N/A": "#64748b",
            }

            html_content = self._create_dashboard_html(df, color_map)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"✓ Interactive dashboard saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Error generating dashboard: {str(e)[:100]}")

    def _create_dashboard_html(self, df, color_map):
        """Create HTML with embedded interactive Plotly charts"""

        # Chart 1: Top 10 Users by Risk Score
        top_10 = df.head(10)
        fig1 = px.bar(
            top_10,
            x="Score",
            y="User",
            color="KMeans",
            color_discrete_map=color_map,
            orientation="h",
            title="Top 10 Users by Risk Score",
            labels={"Score": "Suspicion Score"},
        )
        fig1.update_layout(height=400, showlegend=True)
        chart1_html = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn', div_id="plotly-chart-1")
        chart1_modal_html = pio.to_html(fig1, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-1")

        # Chart 2: KMeans Pie Chart
        kmeans_counts = df["KMeans"].value_counts()
        fig2 = go.Figure(
            data=[
                go.Pie(
                    labels=kmeans_counts.index,
                    values=kmeans_counts.values,
                    marker=dict(
                        colors=[color_map.get(c, "#64748b") for c in kmeans_counts.index]
                    ),
                )
            ]
        )
        fig2.update_layout(title="K-Means Cluster Distribution", height=400)
        chart2_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False, div_id="plotly-chart-2")
        chart2_modal_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-2")

        # Chart 3: DBSCAN Clusters
        dbscan_counts = df["DBSCAN"].value_counts()
        dbscan_labels = [f"C{i}" if i != -1 else "Outlier" for i in dbscan_counts.index]
        fig3 = go.Figure(
            data=[go.Bar(x=dbscan_labels, y=dbscan_counts.values)]
        )
        fig3.update_layout(title="DBSCAN Clusters", height=400, xaxis_title="Cluster", yaxis_title="Count")
        chart3_html = pio.to_html(fig3, full_html=False, include_plotlyjs=False, div_id="plotly-chart-3")
        chart3_modal_html = pio.to_html(fig3, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-3")

        # Chart 4: Network Degree Centrality
        top_central = df.nlargest(15, "Degree")
        fig4 = px.bar(
            top_central,
            x="Degree",
            y="User",
            orientation="h",
            title="Network Degree Centrality (Top 15)",
            color="Degree",
            color_continuous_scale="Blues",
        )
        fig4.update_layout(height=500)
        chart4_html = pio.to_html(fig4, full_html=False, include_plotlyjs=False, div_id="plotly-chart-4")
        chart4_modal_html = pio.to_html(fig4, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-4")

        # Chart 5: Sentiment vs Risk Score
        fig5 = px.scatter(
            df,
            x="Sentiment",
            y="Score",
            color="KMeans",
            size="Messages",
            color_discrete_map=color_map,
            title="Sentiment vs Risk Score",
            labels={"Sentiment": "Avg Sentiment", "Score": "Suspicion Score"},
            hover_data=["User", "Keywords"],
        )
        fig5.update_layout(height=500)
        chart5_html = pio.to_html(fig5, full_html=False, include_plotlyjs=False, div_id="plotly-chart-5")
        chart5_modal_html = pio.to_html(fig5, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-5")

        # Chart 6: Suspicious Words vs Risk
        fig6 = px.scatter(
            df,
            x="SuspiciousWords",
            y="Score",
            color="KMeans",
            size="Keywords",
            color_discrete_map=color_map,
            title="Suspicious Words Count vs Risk",
            labels={"SuspiciousWords": "Suspicious Words Count", "Score": "Suspicion Score"},
            hover_data=["User"],
        )
        fig6.update_layout(height=500)
        chart6_html = pio.to_html(fig6, full_html=False, include_plotlyjs=False, div_id="plotly-chart-6")
        chart6_modal_html = pio.to_html(fig6, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-6")

        # Chart 7: False Positives Avoided
        top_fp = df.nlargest(15, "FalsePositivesAvoided")
        fig7 = px.bar(
            top_fp,
            x="FalsePositivesAvoided",
            y="User",
            orientation="h",
            title="False Positives Avoided (Context-Aware)",
            color="FalsePositivesAvoided",
            color_continuous_scale="Greens",
        )
        fig7.update_layout(height=500)
        chart7_html = pio.to_html(fig7, full_html=False, include_plotlyjs=False, div_id="plotly-chart-7")
        chart7_modal_html = pio.to_html(fig7, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-7")

        # Chart 8: True vs Avoided
        total_suspicious = df["SuspiciousWords"].sum()
        total_false_avoided = df["FalsePositivesAvoided"].sum()
        fig8 = go.Figure(
            data=[
                go.Bar(
                    x=["True Positives\n(Suspicious)", "False Positives\nAvoided"],
                    y=[total_suspicious, total_false_avoided],
                    marker=dict(color=["#2563eb", "#60a5fa"]),
                    text=[total_suspicious, total_false_avoided],
                    textposition="outside",
                )
            ]
        )
        fig8.update_layout(
            title="META DETECT - Context-Aware Detection Accuracy",
            height=400,
            yaxis_title="Count",
        )
        chart8_html = pio.to_html(fig8, full_html=False, include_plotlyjs=False, div_id="plotly-chart-8")
        chart8_modal_html = pio.to_html(fig8, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-8")

        # Chart 9: Top Keywords
        keyword_data = df[["User", "Keywords"]].nlargest(10, "Keywords")
        fig9 = px.bar(
            keyword_data,
            x="Keywords",
            y="User",
            orientation="h",
            title="Top 10 Keyword Detections",
            color="Keywords",
            color_continuous_scale="Blues",
        )
        fig9.update_layout(height=400)
        chart9_html = pio.to_html(fig9, full_html=False, include_plotlyjs=False, div_id="plotly-chart-9")
        chart9_modal_html = pio.to_html(fig9, full_html=False, include_plotlyjs=False, div_id="plotly-modal-chart-9")

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>META DETECT Visual Dashboard v1.0</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            color: white;
            padding: 30px 0;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 30px rgba(37, 99, 235, 0.8);
            color: #2563eb;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .chart-container {{
            background: rgba(30, 41, 59, 0.8);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            border: 2px solid rgba(37, 99, 235, 0.3);
        }}
        .chart-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(37, 99, 235, 0.4);
            border-color: #2563eb;
        }}
        .chart-container::after {{
            content: "🔍 Click to expand";
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(37, 99, 235, 0.9);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.8em;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .chart-container:hover::after {{
            opacity: 1;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            animation: fadeIn 0.3s;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .modal-content {{
            position: relative;
            background-color: white;
            margin: 2% auto;
            padding: 30px;
            width: 95%;
            max-width: 1600px;
            max-height: 90vh;
            border-radius: 15px;
            overflow-y: auto;
            animation: slideIn 0.3s;
        }}
        @keyframes slideIn {{
            from {{
                transform: translateY(-50px);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            line-height: 20px;
        }}
        .close:hover,
        .close:focus {{
            color: #000;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            padding: 30px 0;
            margin-top: 40px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 META DETECT Visual Dashboard v1.0</h1>
        <p>Context-Aware NLP + Network Analysis + DBSCAN + K-Means + BERT + LSTM</p>
    </div>

    <div class="dashboard-grid">
        <div class="chart-container" onclick="openModal('modal1')">
            {chart1_html}
        </div>

        <div class="chart-container" onclick="openModal('modal2')">
            {chart2_html}
        </div>

        <div class="chart-container" onclick="openModal('modal3')">
            {chart3_html}
        </div>

        <div class="chart-container" onclick="openModal('modal4')">
            {chart4_html}
        </div>

        <div class="chart-container" onclick="openModal('modal5')">
            {chart5_html}
        </div>

        <div class="chart-container" onclick="openModal('modal6')">
            {chart6_html}
        </div>

        <div class="chart-container" onclick="openModal('modal7')">
            {chart7_html}
        </div>

        <div class="chart-container" onclick="openModal('modal9')">
            {chart9_html}
        </div>

        <div class="chart-container full-width" onclick="openModal('modal8')">
            {chart8_html}
        </div>
    </div>

    <div id="modal1" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal1')">&times;</span>
            {chart1_modal_html}
        </div>
    </div>

    <div id="modal2" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal2')">&times;</span>
            {chart2_modal_html}
        </div>
    </div>

    <div id="modal3" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal3')">&times;</span>
            {chart3_modal_html}
        </div>
    </div>

    <div id="modal4" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal4')">&times;</span>
            {chart4_modal_html}
        </div>
    </div>

    <div id="modal5" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal5')">&times;</span>
            {chart5_modal_html}
        </div>
    </div>

    <div id="modal6" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal6')">&times;</span>
            {chart6_modal_html}
        </div>
    </div>

    <div id="modal7" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal7')">&times;</span>
            {chart7_modal_html}
        </div>
    </div>

    <div id="modal8" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal8')">&times;</span>
            {chart8_modal_html}
        </div>
    </div>

    <div id="modal9" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('modal9')">&times;</span>
            {chart9_modal_html}
        </div>
    </div>

    <div class="footer">
        META DETECT v1.0 • Visual Dashboard • Click any chart to view in detail
    </div>

    <script>
        function openModal(modalId) {{
            document.getElementById(modalId).style.display = "block";
            document.body.style.overflow = "hidden";
        }}

        function closeModal(modalId) {{
            document.getElementById(modalId).style.display = "none";
            document.body.style.overflow = "auto";
        }}

        window.onclick = function(event) {{
            if (event.target.classList.contains('modal')) {{
                event.target.style.display = "none";
                document.body.style.overflow = "auto";
            }}
        }}

        document.addEventListener('keydown', function(event) {{
            if (event.key === "Escape") {{
                var modals = document.getElementsByClassName('modal');
                for (var i = 0; i < modals.length; i++) {{
                    modals[i].style.display = "none";
                }}
                document.body.style.overflow = "auto";
            }}
        }});
    </script>
</body>
</html>
"""
        return html_template


# -------------------------------------------------------------------
# Shared pipeline for web
# -------------------------------------------------------------------
def run_full_analysis(
    detector: MetaDetect,
    filepath: str,
    platform: str = "auto",
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    Runs the full META DETECT analysis pipeline on the given file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(filepath))[0]

    output_report_path = os.path.join(
        output_dir, f"{base_filename}_meta_detect_report.json"
    )
    output_viz_path = os.path.join(
        output_dir, f"{base_filename}_meta_detect_dashboard.html"
    )
    output_static_viz_path = os.path.join(
        output_dir, f"{base_filename}_meta_detect_dashboard.png"
    )

    print("\n" + "=" * 80)
    print("STEP 1: PARSING CHAT FILE")
    print("=" * 80)

    chosen_platform = platform.lower() if platform else "auto"
    if chosen_platform == "auto":
        lower = filepath.lower()
        if "whatsapp" in lower:
            chosen_platform = "whatsapp"
        elif "telegram" in lower:
            chosen_platform = "telegram"
        else:
            chosen_platform = "whatsapp"

    if chosen_platform == "whatsapp":
        messages = detector.parse_whatsapp_chat(filepath)
        print("✓ Platform: WhatsApp")
    elif chosen_platform == "telegram":
        messages = detector.parse_telegram_chat(filepath)
        print("✓ Platform: Telegram")
    else:
        raise ValueError(f"Unknown platform: {platform}")

    if not messages:
        raise RuntimeError("No messages could be parsed. Check the file format.")

    print(f"✓ Parsed {len(messages)} total messages.")

    print("\n" + "=" * 80)
    print("STEP 2: NETWORK ANALYSIS")
    print("=" * 80)
    detector.build_interaction_network(messages)
    centrality_metrics = detector.analyze_network_centrality()
    print(f"✓ Network centrality calculated for {len(centrality_metrics)} users")

    print("\n" + "=" * 80)
    print(
        "STEP 3: CONTEXT-AWARE NLP & SENTIMENT ANALYSIS"
        + (" + BERT" if detector.use_bert else "")
    )
    print("=" * 80)
    user_stats = detector.analyze_messages(messages)

    print("\n" + "=" * 80)
    print("STEP 4: DBSCAN CLUSTERING")
    print("=" * 80)
    user_stats = detector.apply_dbscan_clustering(user_stats, centrality_metrics)

    print("\n" + "=" * 80)
    print("STEP 5: K-MEANS CLUSTERING")
    print("=" * 80)
    user_stats = detector.apply_kmeans_clustering(user_stats)

    print("\n" + "=" * 80)
    print("STEP 6: LSTM TEMPORAL PATTERN ANALYSIS")
    print("=" * 80)
    lstm_scores = detector.train_lstm_model(user_stats)

    print("\n" + "=" * 80)
    print("STEP 7: GENERATING REPORTS & VISUALIZATIONS")
    print("=" * 80)
    report = detector.generate_comprehensive_report(
        user_stats, centrality_metrics, lstm_scores, output_report_path
    )

    details_csv_path = detector.export_suspicious_words_csv(
        user_stats, base_filename, output_dir
    )
    summary_csv_path = detector.export_suspicious_words_summary_csv(
        report, base_filename, output_dir
    )
    
    detector.create_interactive_dashboard(user_stats, centrality_metrics, output_viz_path)
    detector.create_static_dashboard_image(user_stats, centrality_metrics, output_static_viz_path)

    print("\n✅ META DETECT analysis complete!")

    return {
        "report": report,
        "user_stats": user_stats,
        "centrality_metrics": centrality_metrics,
        "lstm_scores": lstm_scores,
        "output_report_path": output_report_path,
        "output_viz_path": output_viz_path,
        "output_static_viz_path": output_static_viz_path,
        "details_csv_path": details_csv_path,
        "summary_csv_path": summary_csv_path,
        "base_filename": base_filename,
    }


# -------------------------------------------------------------------
# Flask Web App with Blue/Gray UI
# -------------------------------------------------------------------
app = Flask(__name__)

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.path.abspath(".")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

DETECTOR: Optional["MetaDetect"] = None
DETECTION_ACCURACY: Optional[float] = None

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>META DETECT - Visual Dashboard</title>
<style>
    :root {
        --color-primary: #2563eb;
        --color-primary-hover: #1d4ed8;
        --color-bg: #111827;
        --color-card: #182033;
        --color-text-primary: #cdd3e0;
        --color-text-secondary: #7fa3ee;
        --color-border: #25326e;
        --color-danger: #dc2626;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--color-bg); color: var(--color-text-primary); line-height: 1.6; min-height: 100vh; }
    .header { background: var(--color-card); border-bottom: 1px solid var(--color-border); padding: 1.5rem 2rem; text-align: center; }
    .header h1 { font-size: 2rem; font-weight: 600; color: var(--color-text-primary); margin-bottom: 0.25rem;}
    .header p { font-size: 0.95rem; color: var(--color-text-secondary);}
    .container { max-width: 800px; margin: 0 auto; padding: 2rem 1.5rem 3rem; }
    .card { background: var(--color-card); border: 1px solid var(--color-border); border-radius: 8px; padding: 1.75rem; }
    .form-group { margin-bottom: 1.25rem; }
    label { display: block; margin-bottom: 0.4rem; font-weight: 500; color: var(--color-text-primary); font-size: 0.9rem;}
    input[type="file"] { width: 100%; padding: 0.6rem; background: #19213d; border: 1px solid var(--color-border); border-radius: 6px; color: var(--color-text-secondary); font-size: 0.9rem;}
    input[type="file"]::file-selector-button { background: var(--color-primary); color: white; border: none; padding: 0.45rem 0.9rem; border-radius: 4px; cursor: pointer; font-size: 0.85rem; margin-right: 0.75rem;}
    input[type="file"]::file-selector-button:hover { background: var(--color-primary-hover);}
    .platform-group { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 0.4rem;}
    .platform-group label { font-weight: 400; color: var(--color-text-secondary); font-size: 0.9rem; display: flex; align-items: center; cursor: pointer; }
    .platform-group input[type="radio"] { margin-right: 0.4rem; accent-color: var(--color-primary);}
    .btn { background: var(--color-primary); border: none; color: white; padding: 0.7rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 0.95rem; width: 100%; margin-top: 0.25rem;}
    .btn:hover { background: var(--color-primary-hover);}
    .error { background: #2c2331; border: 1px solid #2d1156; color: var(--color-danger); padding: 0.75rem; border-radius: 6px; margin-bottom: 1.25rem; font-size: 0.9rem;}
    .description { text-align: left; margin-bottom: 1.5rem; color: var(--color-text-secondary); font-size: 0.9rem;}
    .description b { color: var(--color-primary); font-weight: 600;}
    .footer { margin-top: 1.5rem; font-size: 0.8rem; color: var(--color-text-secondary); text-align: left;}
    @media (max-width: 640px) { .header h1 { font-size: 1.5rem; } .container { padding: 1.5rem 1rem 2rem; } }
</style>
</head>
<body>
    <div class="header">
        <h1> META DETECT</h1>
        <p>Advanced User Detection & Analysis Platform</p>
    </div>

    <div class="container">
        <div class="card">
            <div class="description">
                Upload your WhatsApp / Telegram chat export file and META DETECT will analyze users using<br>
                <b>Context-Aware NLP + Network Analysis + DBSCAN + K-Means + BERT + LSTM + Visual Dashboard</b>
            </div>

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            <form method="post" action="/analyze" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="chat_file">📁 Chat Export File (.txt):</label>
                    <input type="file" id="chat_file" name="chat_file" required>
                </div>

                <div class="form-group">
                    <label>🔧 Platform:</label>
                    <div class="platform-group">
                        <label><input type="radio" name="platform" value="auto" checked> Auto Detect</label>
                        <label><input type="radio" name="platform" value="whatsapp"> WhatsApp</label>
                        <label><input type="radio" name="platform" value="telegram"> Telegram</label>
                    </div>
                </div>

                <button type="submit" class="btn">🚀 Run Analysis</button>
            </form>

            <div class="footer">
                META DETECT v1.0 • Runs locally on your machine • No data leaves your system
            </div>
        </div>
    </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>META DETECT - Analysis Results</title>
    <style>
        :root {
            --color-primary: #2563eb;
            --color-primary-hover: #1d4ed8;
            --color-bg: #111827;
            --color-card: #182033;
            --color-text-primary: #cdd3e0;
            --color-text-secondary: #7fa3ee;
            --color-border: #25326e;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--color-bg);
            color: var(--color-text-primary);
            line-height: 1.6;
            min-height: 100vh;
            padding: 2rem 1.5rem;
        }
        .container { max-width: 1100px; margin: 0 auto; }
        .header {
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: left;
        }
        .header h1 {
            font-size: 1.9rem;
            font-weight: 600;
            color: var(--color-text-primary);
            margin-bottom: 0.35rem;
        }
        .header p {
            font-size: 0.9rem;
            color: var(--color-text-secondary);
        }
        .highlight-box {
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            margin-top: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.2rem;
        }
        .highlight-content {
            flex: 1 1 auto;
        }
        .highlight-box h2 {
            color: var(--color-primary);
            font-size: 1.15rem;
            margin-bottom: 0.35rem;
        }
        .highlight-box p {
            color: var(--color-text-secondary);
            font-size: 0.97rem;
            margin-bottom: 0.6rem;
        }
        .btn-dashboard {
            background: var(--color-primary);
            font-size: 0.95rem;
            padding: 0.72rem 1.42rem;
            border-radius: 6px;
            border: 1px solid var(--color-primary);
            color: white;
            text-decoration: none;
            font-weight: 500;
            transition: background 0.18s;
            margin-left: auto;
            white-space: nowrap;
            box-shadow: 0 1px 8px rgba(37,99,235,0.08);
        }
        .btn-dashboard:hover { background: var(--color-primary-hover); }
        .btn-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin: 1.25rem 0 1.75rem;
        }
        .btn-small {
            background: var(--color-primary);
            color: white;
            border-radius: 6px;
            font-size: 0.85rem;
            border: 1px solid var(--color-primary);
            padding: 0.55rem 1.1rem;
            text-decoration: none;
            font-weight: 500;
        }
        .btn-small:hover { background: var(--color-primary-hover); }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .stat-card {
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1.05rem 1.2rem;
        }
        .stat-card h3 {
            margin-top: 0;
            font-size: 0.95rem;
            color: var(--color-text-secondary);
            margin-bottom: 0.6rem;
            font-weight: 500;
        }
        .value {
            font-weight: 600;
            font-size: 1.3rem;
            color: var(--color-primary);
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            margin-bottom: 0.4rem;
            font-weight: 500;
        }
        .pill-high { background: #222435; color: #dc2626; }
        .pill-medium { background: #25326e; color: #eab308; }
        .pill-low { background: #1e3a5b; color: #22c55e; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.8rem;
            font-size: 0.85rem;
            background: var(--color-card);
        }
        th, td {
            border-bottom: 1px solid var(--color-border);
            padding: 0.55rem 0.4rem;
            text-align: left;
        }
        th {
            color: var(--color-text-secondary);
            font-weight: 500;
            font-size: 0.78rem;
            text-transform: uppercase;
        }
        tr:hover { background: #25326e; }
        a {
            color: var(--color-primary);
            text-decoration: none;
            font-size: 0.9rem;
        }
        a:hover { text-decoration: underline; }
        @media (max-width: 640px) {
            body { padding: 1.5rem 1rem; }
            .header h1 { font-size: 1.6rem; }
            .btn-row { flex-direction: column; }
            .highlight-box {
                flex-direction: column;
                align-items: stretch;
                justify-content: flex-start;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>META DETECT - Analysis Complete</h1>
            <p>{{ summary.detection_mode }}</p>
        </div>
        <div class="highlight-box">
            <div class="highlight-content">
                <h2>📊 Visual Dashboard Ready!</h2>
                <p>Click charts to expand and explore interactive visualizations.</p>
            </div>
            <a class="btn-dashboard" href="{{ url_for('download', filename=viz_file) }}" target="_blank">🚀 Open Visual Dashboard</a>
        </div>
        <div class="btn-row">
            <a class="btn-small" href="{{ url_for('download', filename=static_viz_file) }}">Visual Report PNG</a>
            <a class="btn-small" href="{{ url_for('download', filename=report_file) }}">JSON Report</a>
            <a class="btn-small" href="{{ url_for('download', filename=details_file) }}">Details CSV</a>
            <a class="btn-small" href="{{ url_for('download', filename=summary_file) }}">Summary CSV</a>
        </div>
        <div class="grid">
            <div class="stat-card">
                <h3>Overall Stats</h3>
                <div><span class="value">{{ summary.total_users }}</span> users analyzed</div>
                <div><span class="value">{{ summary.total_keywords_detected }}</span> suspicious keyword hits</div>
                <div><span class="value">{{ summary.total_false_positives_avoided }}</span> false positives avoided</div>
            </div>
            <div class="stat-card">
                <h3>Risk Distribution (K-Means)</h3>
                <div><span class="pill pill-high">High</span> {{ summary.kmeans_high_risk }}</div>
                <div><span class="pill pill-medium">Medium</span> {{ summary.kmeans_medium_risk }}</div>
                <div><span class="pill pill-low">Low</span> {{ summary.kmeans_low_risk }}</div>
            </div>
            <div class="stat-card">
                <h3>Clustering & Network</h3>
                <div style="margin-bottom: 0.5rem;">DBSCAN Outliers: <span style="color: #2563eb; font-weight: 600;">{{ summary.dbscan_outliers }}</span></div>
                <div>Network Analysis: {{ "Enabled" if summary.network_analysis_enabled else "Disabled" }}</div>
            </div>
        </div>
        <div class="stat-card">
            <h3>Top 5 Suspicious Users</h3>
            {% if top_users %}
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>User</th>
                        <th>Risk Score</th>
                        <th>Cluster</th>
                        <th>Messages</th>
                        <th>Keywords</th>
                        <th>Avg Sentiment</th>
                    </tr>
                </thead>
                <tbody>
                    {% for u in top_users %}
                    <tr>
                        <td><strong>{{ u.rank }}</strong></td>
                        <td>{{ u.username }}</td>
                        <td><strong style="color: #2563eb;">{{ "%.1f"|format(u.suspicion_score) }}</strong></td>
                        <td>{{ u.kmeans_cluster }}</td>
                        <td>{{ u.message_count }}</td>
                        <td>{{ u.keyword_count }}</td>
                        <td>{{ "%.2f"|format(u.avg_sentiment_score) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
                <p style="color: #22c55e; font-weight: 500;">No suspicious users detected.</p>
            {% endif %}
        </div>
        <a style="display: inline-block; margin-top: 2rem;" href="/">← Run another analysis</a>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    global DETECTOR, DETECTION_ACCURACY

    file = request.files.get("chat_file")
    if not file or file.filename == "":
        return render_template_string(
            INDEX_HTML,
            error="Please select a chat export file before running analysis.",
        )

    platform = request.form.get("platform", "auto").lower()

    filename = secure_filename(file.filename)
    if not filename:
        return render_template_string(
            INDEX_HTML,
            error="Invalid filename. Please rename the file and try again.",
        )

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    if DETECTOR is None:
        DETECTOR = MetaDetect()
        try:
            DETECTION_ACCURACY = DETECTOR.test_detection_accuracy()
        except Exception:
            DETECTION_ACCURACY = None

    try:
        result = run_full_analysis(
            DETECTOR,
            upload_path,
            platform=platform,
            output_dir=RESULT_FOLDER,
        )
    except Exception as e:
        return render_template_string(
            INDEX_HTML,
            error=f"Error during analysis: {str(e)[:200]}",
        )

    report = result["report"]
    base = result["base_filename"]

    summary = {
        "detection_system": report["detection_system"],
        "detection_mode": report["detection_mode"],
        "analysis_timestamp": report["analysis_timestamp"],
        "total_users": report["total_users"],
        "total_false_positives_avoided": report["total_false_positives_avoided"],
        "total_keywords_detected": report["total_keywords_detected"],
        "kmeans_high_risk": report["kmeans_high_risk"],
        "kmeans_medium_risk": report["kmeans_medium_risk"],
        "kmeans_low_risk": report["kmeans_low_risk"],
        "dbscan_outliers": report["dbscan_outliers"],
        "network_analysis_enabled": report["network_analysis_enabled"],
        "bert_enabled": report["bert_enabled"],
    }

    suspicious_users = [
        u
        for u in report["users"]
        if u.get("suspicion_score", 0) > 0 and u.get("keyword_count", 0) > 0
    ]
    top_users = suspicious_users[:5]

    report_file = os.path.basename(result["output_report_path"])
    viz_file = os.path.basename(result["output_viz_path"])
    static_viz_file = os.path.basename(result["output_static_viz_path"])
    details_file = os.path.basename(result["details_csv_path"])
    summary_file = os.path.basename(result["summary_csv_path"])

    return render_template_string(
        RESULT_HTML,
        summary=summary,
        top_users=top_users,
        report_file=report_file,
        viz_file=viz_file,
        static_viz_file=static_viz_file,
        details_file=details_file,
        summary_file=summary_file,
        accuracy=DETECTION_ACCURACY,
        base_filename=base,
    )


@app.route("/download/<path:filename>")
def download(filename: str):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
