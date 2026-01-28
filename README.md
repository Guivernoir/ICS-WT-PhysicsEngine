# Water Treatment CSTR Physics Simulator

An industrial-grade digital twin of a **Continuous Stirred-Tank Reactor (CSTR)** designed for water treatment simulation. This project integrates a high-fidelity physics engine with a hardened **Modbus TCP** interface, allowing for real-time monitoring and control via PLCs, SCADA systems, or custom applications.

---

## 🚀 Overview

This simulator models the chemical and physical dynamics of a multi-zone reactor. It doesn't just "output numbers", it solves the underlying differential equations for mass balance, chemical equilibrium, and sensor dynamics in real-time.

### Key Features

* **Physics Engine**: Multi-zone CSTR model accounting for spatial gradients, mixing, and temperature-dependent kinetics.
* **Aqueous Chemistry**: Real-time pH calculation using Newton-Raphson iteration on the carbonate buffer system and charge balance.
* **Modbus TCP**: Powered by `pymodbus 3.11.4`, providing a stable asynchronous server for industrial interoperability.
* **Realistic Sensors**: Sensors include physical characteristics like membrane hydration (warm-up times), noise, drift, and response time ().

---

## 🛠 Tech Stack

* **Language**: Python 3.14.2
* **Physics/Math**: NumPy, SciPy
* **Protocol**: Pymodbus (Async TCP)
* **Orchestration**: Python `asyncio` & `threading`

---

## 📊 Process Values & Constants

The simulator exposes several critical process variables and configuration constants via Modbus registers.

### Dynamic Process Data (Read-Only)

| Parameter | Description | Typical Range |
| --- | --- | --- |
| **pH Inlet** | Incoming water acidity | 6.5 - 8.5 pH |
| **pH Outlet** | Treated water acidity | 6.0 - 9.0 pH |
| **Chlorine Out** | Residual disinfection level | 0.2 - 4.0 mg/L |
| **System Flow** | Current throughput | 0.0 - 20.0 L/min |

### Simulation Constants (Holding Registers)

| Constant | Default | Description |
| --- | --- | --- |
| `warmup_time_s` | 1800s | Sensor stabilization period |
| `v_total` | 500L | Total reactor capacity |
| `alkalinity` | 100 mg/L | Water buffering capacity () |

---

## 📦 Installation

1. **Ensure Python 3.14.2+ is installed.**
2. **Clone the repository**:
```bash
git clone https://github.com/Guivernoir/water-treatment-sim.git
cd water-treatment-sim

```


3. **Install dependencies**:
```bash
pip install .

```



---

## 🏃 Usage

Start the simulation and the Modbus server:

```bash
python __main__.py

```

By default, the Modbus server will listen on `0.0.0.0:5020`. You can verify the data using any Modbus client:

```bash
# Example using pymodbus console
python -m pymodbus.console tcp --host 127.0.0.1 --port 5020

```

---

## 🧪 Physics Architecture

The simulation follows a modular approach:

1. **`reactor.py`**: Handles the spatial distribution and mass transport between zones.
2. **`chemistry.py`**: Solves the non-linear equations for chemical species and pH.
3. **`ph_sensor.py`**: Adds the "Real-World" layer of noise and instrument limitations.
4. **`slave.py`**: Bridges the physics objects to the Modbus data blocks.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
