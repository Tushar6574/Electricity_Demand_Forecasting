Skill: Grid Operational Response
Goal: Categorize forecast demand and trigger industrial management protocols.

SOP
1. Threshold Analysis: Analyze the `predicted_demand` column for the next 24-hour window.
2. Categorization Logic:
   - High Demand (>90% Capacity): Trigger "Peak Shaving" protocol.
   - Low Demand (<40% Capacity): Trigger "Economic Dispatch & Battery Charging".
   - Normal: Maintain standard load balancing.
3. Hardware Simulation: Log the required battery discharge/charge rate (MW) based on the variance from the 50% load baseline.
4. Notification: Output a summary of "Actionable Triggers" for the grid operator.
