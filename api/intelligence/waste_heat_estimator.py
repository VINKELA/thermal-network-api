import math

class WasteHeatEstimator:
    """
    Estimates monthly waste heat potential (kWh) based on the methodology 
    by F. Albert (2025).
    """

    @staticmethod
    def estimate_monthly_heat(building, month_idx):
        """
        Main entry point for calculating waste heat.
        :param building: Building model instance
        :param month_idx: 1 (Jan) to 12 (Dec)
        :return: Estimated waste heat in kWh for that month
        """
        # Safely get source type
        source_type = getattr(building, 'waste_heat_source_type', None)
        if not source_type:
            return 0.0

        # Get Outdoor Air Temp (OAT) for the month
        oat = WasteHeatEstimator._get_ontario_avg_temp(month_idx) 

        if source_type == 'data_center': 
            return WasteHeatEstimator._calc_data_centre(building, oat)
        elif source_type == 'grocery_store':
            return WasteHeatEstimator._calc_grocery_store(building, oat)
        elif source_type == 'cold_storage':
            return WasteHeatEstimator._calc_cold_storage(building, oat)
        elif source_type == 'ice_arena':
            return WasteHeatEstimator._calc_ice_arena(building, month_idx)
        elif source_type == 'space_cooling':
            return WasteHeatEstimator._calc_space_cooling(building, oat, month_idx)
        elif source_type == 'wastewater':
            return WasteHeatEstimator._calc_wastewater(building, oat)
        
        return 0.0

    # --- 1. DATA CENTRES [Section 4.2.2] ---
    @staticmethod
    def _calc_data_centre(building, oat):
        """
        Assumption: 90% of total electrical consumption is rejected as heat.
        Average operation is 70% of max capacity.
        """
        # Try to get actual consumption if available
        monthly_kwh = 320
        
        # FALLBACK: Estimate based on Floor Area if consumption is missing
        # Typical Data Centre EUI is extremely high (e.g., 1000+ kWh/m2/yr)
        if not monthly_kwh:
            floor_area = getattr(building, 'floor_area', 0) or 1000
            # Estimate: 1500 kWh/m2/year * floor_area / 12 months
            monthly_kwh = (1500 * floor_area) / 12

        # 90% recovery factor (Electrical energy -> Heat)
        print(f"Data Centre: Estimated monthly electricity = {monthly_kwh:.2f} kWh, OAT = {oat}C")
        return monthly_kwh

    # --- 2. GROCERY STORES [Section 4.2.3] ---
    @staticmethod
    def _calc_grocery_store(building, oat):
        """
        Formula: Q_residual = W_elec_refrig * (1 + COP)
        """
        gfa = getattr(building, 'floor_area', 3500) or 3500
        
        # Constants from Thesis
        eui_annual_gj = 3.4 
        eui_annual_kwh = eui_annual_gj * 277.778  # Correctly converted GJ to kWh
        percent_refrig = 0.30 
        
        # Monthly electricity for refrigeration
        # BUG FIX: Removed the rogue * 1000000 multiplier
        w_elec_refrig = (eui_annual_kwh * gfa * percent_refrig) / 12 

        # COP Curve Approximation (CO2 Systems)
        # COP decreases as OAT increases
        cop = max(1.5, min(4.75, 4.4 - (0.1 * oat))) 

        return w_elec_refrig * (1 + cop)

    # --- 3. COLD STORAGE [Section 4.2.4] ---
    @staticmethod
    def _calc_cold_storage(building, oat):
        """
        Formula: Q_residual = W_elec_refrig * (1 + COP)
        """
        area = getattr(building, 'floor_area', 1000) or 1000
        # Estimate volume if missing (assume 10m height for warehouse)
        volume = getattr(building, 'volume_m3', None) or (area * 10) 
        
        annual_elec_consumption = 35 * volume # 35 kWh/m3/year
        w_elec_refrig_monthly = (annual_elec_consumption * 0.70) / 12

        # COP Average ~3.0 
        cop = 3.0 
        
        return w_elec_refrig_monthly * (1 + cop)

    # --- 4. ICE ARENAS [Section 4.2.5] ---
    @staticmethod
    def _calc_ice_arena(building, month_idx):
        """
        Formula: Q_residual = W_elec_refrig * (1 + COP)
        """
        is_year_round = getattr(building, 'is_year_round_operation', False)
        
        # Seasonal: Sept (9) to April (4)
        operating_months = [1, 2, 3, 4, 9, 10, 11, 12] 
        
        if not is_year_round and month_idx not in operating_months:
            return 0.0

        gfa = getattr(building, 'floor_area', 3900) or 3900
        eui_annual_gj = 1.64
        eui_annual_kwh = eui_annual_gj * 277.778
        
        # 48% is Elec, 65% of that is Refrigeration
        annual_refrig_load = eui_annual_kwh * gfa * 0.48 * 0.65
        
        # Distribute based on operating months
        months_active = 12 if is_year_round else 8
        monthly_refrig_load = annual_refrig_load / months_active

        # Ammonia system COP
        cop = 2.3 

        return monthly_refrig_load * (1 + cop)

    # --- 5. SPACE COOLING [Section 4.2.6] ---
    @staticmethod
    def _calc_space_cooling(building, oat, month_idx):
        """
        Estimates heat rejected by building chillers.
        Method: Cooling Degree Days (CDD) estimation relative to balance point.
        Large buildings often require cooling even at 10C OAT due to internal gains.
        """
        # 1. Determine Balance Point Temperature
        # Large commercial buildings ~10C, Residential ~18C
        balance_point = 10.0 

        # 2. Check if cooling is needed
        if oat <= balance_point:
            return 0.0

        # 3. Estimate Cooling Load
        # Simplified CDD approximation for the month:
        # (Avg Temp - Balance Point) * Days in Month * 24 Hours
        delta_t = oat - balance_point
        hours_in_month = 730 # Average
        cooling_degree_hours = delta_t * hours_in_month

        gfa = getattr(building, 'floor_area', 5000) or 5000
        
        # Cooling Intensity Factor (Approximate)
        # 0.05 kW/m2 peak load roughly
        # We model this as: Load = k * Area * DeltaT
        # k factor derived from typical EUI for cooling (~30 kWh/m2/yr)
        # Let's use a simpler EUI approach distributed by CDD
        
        annual_cooling_eui_kwh_m2 = 40.0 # kWh/m2/year for office
        est_total_annual_cooling = annual_cooling_eui_kwh_m2 * gfa

        # Calculate weighting factor for this month based on Temp
        # We assume summer is roughly May-Sept. 
        # But specifically, we use the DeltaT as the weight.
        # Max DeltaT in Ontario (July 22C vs 10C base) = 12
        # Sum of positive DeltaTs across year ~ 50 (rough integral)
        
        weight = delta_t / 50.0 
        # Clamp weight reasonable bounds (e.g. July might be 25% of annual load)
        monthly_cooling_load = est_total_annual_cooling * weight

        # 4. Convert to Rejected Heat
        # Q_rejected = Q_cooling_load * (1 + 1/COP)
        # Chiller COP ~ 4.0
        cop = 4.0
        q_rejected = monthly_cooling_load * (1 + (1/cop))
        
        return max(0.0, q_rejected)
    
    # --- 6. WASTEWATER [Bonus/Placeholder] ---
    @staticmethod
    def _calc_wastewater(building, oat):
        """
        Placeholder for Wastewater Treatment Plant heat recovery.
        Q = m_dot * cp * DeltaT
        """
        return 0.0

    @staticmethod
    def _get_ontario_avg_temp(month):
        """
        Returns average outdoor air temperature (C) for Southern Ontario 
        based on population weighted profile.
        """
        temps = {
            1: -6.0, 2: -5.0, 3: -1.0, 4: 7.0, 5: 14.0, 6: 19.0,
            7: 22.0, 8: 21.0, 9: 17.0, 10: 10.0, 11: 4.0, 12: -2.0
        }
        return temps.get(month, 10.0)