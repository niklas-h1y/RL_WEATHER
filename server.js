// Speichere diese Datei als: server.js
const http = require('http');
const fs = require('fs');

const STATE_FILE = 'weather_rl_state.json';
const PORT = process.env.PORT || 3000;

class ContinuousWeatherRLAgent {
    constructor(learningRate = 0.01) {
        this.lr = learningRate;
        this.weights = { temp_lag1: 0.5, temp_lag2: 0.3, bias: 0.0 };
        this.score = 0;
        this.total_steps_trained = 0;
        this.current_prediction = 0;
        this.current_actual = 0;
        this.loadSavedState();
    }

    loadSavedState() {
        if (fs.existsSync(STATE_FILE)) {
            try {
                const data = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
                this.weights = data.weights || this.weights;
                this.score = data.score !== undefined ? data.score : this.score;
                this.total_steps_trained = data.total_steps_trained !== undefined ? data.total_steps_trained : this.total_steps_trained;
            } catch (e) {
                console.log("Starte frisches Training.");
            }
        }
    }

    saveState() {
        try {
            fs.writeFileSync(STATE_FILE, JSON.stringify({
                weights: this.weights, this.score: this.score, total_steps_trained: this.total_steps_trained
            }, null, 4));
        } catch (e) {
            console.log("Speicherfehler.");
        }
    }

    trainStep(t1, t2, actualNextTemp) {
        this.current_actual = actualNextTemp;
        this.current_prediction = Math.round(((t1 * this.weights.temp_lag1) + (t2 * this.weights.temp_lag2) + this.weights.bias) * 100) / 100;
        
        const error = actualNextTemp - this.current_prediction;
        const absError = Math.abs(error);

        this.score += (absError < 0.5) ? 20 : (absError < 1.5 ? 5 : -15);
        this.total_steps_trained += 1;

        // KI Synapsen anpassen (Lernen)
        this.weights.temp_lag1 += this.lr * error * t1 * 0.01;
        this.weights.temp_lag2 += this.lr * error * t2 * 0.01;
        this.weights.bias += this.lr * error * 0.1;

        this.saveState();
    }
}

const agent = new ContinuousWeatherRLAgent();
const mockTemps = [14.5, 15.0, 16.5, 18.2, 20.1, 22.4, 24.1, 25.0, 24.3, 22.1, 18.0, 15.5];
let idx = 2;

// Endlose Trainingsschleife (läuft autark alle 3 Sekunden)
setInterval(() => {
    agent.trainStep(mockTemps[idx-1], mockTemps[idx-2], mockTemps[idx]);
    idx = (idx + 1) % mockTemps.length;
    if (idx < 2) idx = 2;
}, 3000);

// Webserver, damit dein Handy die Daten abrufen kann
const server = http.createServer((req, res) => {
    // CORS-Zertifikat erlauben, damit HTML vom Handy zugreifen darf
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');

    // Gibt den aktuellen KI-Zustand als JSON aus
    const dataResponse = {
        total_steps: agent.total_steps_trained,
        score: agent.score,
        actual: agent.current_actual,
        prediction: agent.current_prediction,
        weights: agent.weights
    };
    
    res.end(JSON.stringify(dataResponse));
});

server.listen(PORT, () => console.log(`Server läuft auf Port ${PORT}`));
