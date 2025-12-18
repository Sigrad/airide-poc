
# AIRide PoC: Crowd Flow Prediction 🎢

Streamlit: https://airide-poc-cteaq2kf3citnpxgux8sz8.streamlit.app/

Proof of Concept für die prädiktive Analyse von Wartezeiten im Europa-Park Rust mittels Machine Learning (Random Forest) und meteorologischen Indizes (HCI).

**Projektarbeit HS 25 | Sakir Dönmez**

## 🚀 Features
* **Live Data Harvesting:** Automatische Sammlung von Wartezeiten (Queue-Times API) und Wetterdaten (OpenMeteo API).
* **Feature Engineering:** Berechnung des *Holiday Climate Index (HCI)* und Berücksichtigung von Ferien im Dreiländereck (DE/CH/FR).
* **Machine Learning:** Random Forest Regressor zur Vorhersage.
* **Interactive Dashboard:** Streamlit-App zur Live-Analyse und Simulation ("Was-wäre-wenn").

## 🛠 Installation & Start

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/Sigrad/airide-poc.git](https://github.com/Sigrad/airide-poc.git)
    cd airide-poc
    ```

2.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **App starten:**
    ```bash
    streamlit run streamlit_app.py
    ```

4.  **(Optional) Datensammler starten:**
    ```bash
    python data_harvester.py
    ```

## 📊 Tech Stack
* **Python:** Core Logic
* **Streamlit:** Frontend / Dashboard
* **Scikit-Learn:** Random Forest Model
* **Pandas/NumPy:** Data Processing
* **OpenMeteo & Queue-Times:** Data Sources
