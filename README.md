# PraanAI 🏥
**An AI-powered emergency triage assistant for hospital casualty wards**

*प्राण — Sanskrit for "Vital Life Force"*

---

## Why we built this

Anyone who has been to a government hospital emergency ward in India knows what it looks like — overcrowded, understaffed, and chaotic. Nurses running between 20–30 patients at once. Doctors making split-second decisions based on handwritten notes passed across a room. Patients waiting without anyone knowing how critical their condition actually is.

The triage process — deciding who needs treatment first — takes 15 to 20 minutes per patient when done manually. That's not because anyone is lazy or incompetent. It's just how paper-based, human-relay systems work. But in an emergency ward, 15 minutes can be the difference between recovery and permanent damage.

We built PraanAI to fix that one specific bottleneck.

---

## What it does

A nurse enters the patient's vitals when they arrive — blood pressure, heart rate, temperature, oxygen saturation, respiratory rate, and pain score. That takes about 85 seconds with standard bedside equipment that every hospital already has.

PraanAI then predicts the patient's ESI triage level in under 3 seconds using an XGBoost machine learning model, and instantly updates the doctor's live dashboard. No paper. No relay. No waiting.

The doctor sees a prioritized queue the moment a patient is registered. Highest risk patients are always at the top.

---

## The numbers

| Step | Manual | PraanAI |
|------|--------|---------|
| Vitals collection | 85 sec | 85 sec (same) |
| Decision + queue update | 13–18 min | 3 sec |
| **Total** | **15–20 min** | **under 2 min** |

With 20 patients a day, that's 260 minutes saved daily. Over a year, that adds up to roughly 65 days of doctor-time freed up — time that can go toward actual treatment instead of paperwork and verbal relays.

---

## Pages

| Page | What it does |
|------|-------------|
| `/` | Nurse enters patient vitals and gets instant AI triage result |
| `/dashboard` | Doctor's live queue, sorted by severity, updates in real-time |
| `/search` | Search patients by name, ESI level, date, or verification status |
| `/reports` | Analytics — total patients, ESI breakdown charts, time saved |

---

## One honest thing about the fallback system

Not every hospital has every device available at all times. If a nurse can't measure BP because the machine is down, they can tick "Not available" and PraanAI uses a WHO-standard average value instead.

But we don't pretend that's ideal. If a critical field like blood pressure or oxygen saturation uses a fallback, the dashboard shows a red **"Verify Manually"** warning — because the AI decision in that case may not be accurate, and a doctor should double-check before acting on it.

We think that kind of transparency matters more than pretending the system is perfect.

---

## How to run it

**Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/HealthCare-Project.git
cd HealthCare-Project
```

**Set up the environment**
```bash
py -3.10 -m venv praanai_env
praanai_env\Scripts\activate
pip install -r requirements.txt
```

**Get the dataset**

Download `triage.csv` from [this Kaggle dataset](https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application) and put it in the `dataset/` folder.

**Train the model**
```bash
python train_model.py
```

**Run the app**
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Project structure

```
HealthCare-Project/
├── app.py                  # Flask backend and all API routes
├── train_model.py          # Model training script
├── requirements.txt
├── model/
│   ├── triage_model.pkl
│   └── label_encoder.pkl
├── dataset/
│   └── triage.csv
├── templates/
│   ├── index.html          # Patient intake form
│   ├── dashboard.html      # Live doctor dashboard
│   ├── search.html         # Patient search
│   └── reports.html        # Analytics
└── static/
```

---

## Tech stack

- **Backend** — Python, Flask, Flask-SocketIO
- **AI Model** — XGBoost, scikit-learn, pandas, numpy
- **Frontend** — HTML, CSS, JavaScript
- **Database** — SQLite
- **Real-time updates** — Socket.io
- **Dataset** — Kaggle Emergency Service Triage (1,267 patient records)

---

## Model performance

The model runs on a dataset of 1,267 patients from Kaggle and hits about **65% accuracy** on the test set. That's a reasonable baseline for a competition MVP — but we won't oversell it. Real hospital deployment would need 10,000+ records and proper clinical validation before anyone should rely on it for actual treatment decisions.

The model predicts one of four ESI levels:

- **ESI 2** — Emergent, treat within 10 minutes
- **ESI 3** — Urgent, treat within 30 minutes
- **ESI 4** — Less urgent, within 1 hour
- **ESI 5** — Non-urgent, within 2 hours

---

## What we'd build next

A few things we scoped out but didn't have time for:

- Continuous vitals monitoring via IoT devices instead of manual entry
- Telugu, Hindi, and Tamil language support for nurses
- A mobile app so nurses aren't tied to a desktop
- Integration with existing hospital HIS systems
- Retraining on larger, real hospital datasets

---

## Team

Five people. One week. Built for our college hackathon under the AI for Healthcare theme.

---

*Built with care by Team PraanAI*
*प्राण — Vital Life Force*
