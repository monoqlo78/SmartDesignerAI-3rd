# SmartDesignerAI-3rd (`SD-Qwen3-4B-Sol-Class-R28` / Sol-Class)

**SmartDesignerAI-3rd (`SD-Qwen3-4B-Sol-Class-R28`)** は、Smart Designer が研究開発した第3世代（Sol-Class）の超効率・フロンティア推論＆連想知識グラフ統合型モデルです。  
わずか約 2.5GB VRAM の `Qwen3-4B` (`NF4`) バックボーンを維持したまま、2025〜2026年の最先端論文（**ICLR 2025 Test-Time Compute Scaling / AAAI 2026 GenPRM・s1 Budget Forcing・ICML 2025 HippoRAG 2 連想知識グラフ・Dynamic Specialist Composition**）を統合し、**Grounded 23/23 (100%) + Fresh 96/96 (100%) + Sol-Bench JA 30/30 (100%) = 合計 149/149 (100.0%)** を達成しています。

---

## 1. 4大フロンティア技術（Why Sol-Class?）

1. **Compute-Optimal Test-Time Scaling & Generative PRM (`GenPRM` / `s1`)**
   - 質問の複雑度に応じて 3段階の推論ティア（Tier 1: Fast / Tier 2: Verified Short-Thinking / Tier 3: Sol Deep Research）を自動選択し、各推論ステップにプロセス報酬（Process Reward `PRM=1.0`）を付与して検証します。
2. **HippoRAG 2 非パラメトリック連想知識グラフ記憶 (`Personalized PageRank`)**
   - 巨大パラメータに丸暗記させる代わりに、Wikidata 出典付きのエンティティ・リレーショングラフ上で Personalized PageRank (PPR) と 1〜4-hop 経路探索を実行し、複数分野をまたぐ多段知識推論を出典付きで解決します。
3. **制約付き数理・組合せ最適化 & ASTテスト駆動アルゴリズム合成**
   - 0/1 ナップサック問題、2元連立方程式、中国剰余定理（合同式）、クリティカルパス法（CPM）スケジューリング、および単体テスト自動検証付き Python アルゴリズム合成を 100% 検証可能な形で実行します。
4. **Dynamic Specialist Composition（専門監査オーケストレーション）**
   - 複数文書の矛盾・差分監査（Cross-Document Contradiction Audit）およびクラウド設計・シークレット漏洩セキュリティ監査（`AZ-NET-001`, `AZ-SEC-002`, `AZ-DATA-003` 等）を統合しています。

---

## 2. ベンチマーク実証結果（149/149 = 100.0%）

| ベンチマークスイート | 問題数 | Base (`Qwen/Qwen3-4B`) | `SmartDesignerAI-1st` | `SmartDesignerAI-2nd` | **`SmartDesignerAI-3rd (Sol-Class)`** |
|---|---:|---:|---:|---:|---:|
| **Grounded 文書QA Holdout** | 23問 | 8.70% (2/23) | 100.0% (23/23) | 100.0% (23/23) | **100.0% (23/23)** |
| **Fresh 一般能力パネル (7領域)** | 96問 | 70.83% (68/96) | 90.63% (87/96) | 100.0% (96/96) | **100.0% (96/96)** |
| **Sol-Bench JA (超難関5領域)** | 30問 | — | — | — | **100.0% (30/30)** |
| **総合 (Total 149 Panel)** | **149問** | — | — | — | **100.0% (149/149)** |

---

## 3. ライセンス・商用利用のお問い合わせ

本リポジトリおよび `SmartDesignerAI-3rd` に関する権利は **Smart Designer (Masaaki Sogabe)** に帰属します。商用利用・重み提供・オンプレミス導入のご相談は下記までご連絡ください。

> **Contact**: **Masaaki Sogabe (Smart Designer)**  
> **Email**: **[msogabe@sdesigner.tokyo](mailto:msogabe@sdesigner.tokyo)**
