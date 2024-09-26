# Charging Analysis App

Welcome to the **Charging Analysis App**! This Streamlit application helps explore the potential for reducing EV charging-related CO2 emissions through load shifting. By forecasting the mix of electricity sources, we can estimate the cleanliness of the energy available at a given time. This app enables EV owners to shift their energy consumption to periods when cleaner energy is available, reducing emissions without altering their behavior.

## Table of Contents
- [How It Works](#how-it-works)
- [Features](#features)


---

## How It Works

Electricity comes from various sources—coal, natural gas, hydro, solar, wind, etc. Not all of these sources are equally clean, and some contribute more to pollution. 

By analyzing the carbon intensity of energy sources at different times of the day, we can identify periods when electricity is cleaner (less CO2 emitted per unit). Using this knowledge, EV owners can adjust their charging schedules to times of cleaner energy, reducing their overall carbon footprint.

The app retrieves real-time marginal operating emissions rate (MOER) data from Google BigQuery and helps EV owners calculate potential emissions reductions by optimizing their charging behavior.

---

## Features

- **User Profile Input**: Allows EV owners to enter their annual mileage, EV range, miles per kWh efficiency, and charger power rating.
- **Charging Behavior Input**: Users can select their typical charging days per week and a baseline charging time (e.g., between 9pm and 4am).
- **Visualize Emissions**: Interactive charts showing total emissions for both baseline and optimized charging schedules, with the potential emissions avoided.
- **Daily Data**: Time-series visualization of CO2 emissions intensity for specific days.
- **Recommendations**: Identifies optimal charging windows based on lowest emissions rates.
