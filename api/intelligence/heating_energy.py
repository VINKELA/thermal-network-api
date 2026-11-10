"""Versioned Canadian benchmark anchors and explicit sensitivity assumptions."""
SHEU_SOURCE = 'https://oee.rncan.gc.ca/publications/statistics/sheu/2015/index.cfm'
SCIEU_SOURCE = 'https://www150.statcan.gc.ca/n1/daily-quotidien/220805/t002c-eng.htm'
RES_SHARE_SOURCE = 'https://natural-resources.canada.ca/energy-efficiency/energy-star/products/list-certified-products/water-heaters'
COM_SHARE_SOURCE = 'https://natural-resources.canada.ca/energy-efficiency/product-energy-ratings/industrialcommercial-equipment'
ENERGY_SOURCES = [SHEU_SOURCE, SCIEU_SOURCE, RES_SHARE_SOURCE, COM_SHARE_SOURCE]
ENERGY_PRESETS = {
    'residential': dict(label='Residential national average', intensity_kwh_m2=.51 / .0036, year=2015, source=SHEU_SOURCE),
    'commercial_average': dict(label='Commercial/institutional average', intensity_kwh_m2=1.31 / .0036, year=2019, source=SCIEU_SOURCE),
    'office': dict(label='Office (non-medical)', intensity_kwh_m2=1.04 / .0036, year=2019, source=SCIEU_SOURCE),
    'retail': dict(label='Retail (non-food)', intensity_kwh_m2=1.04 / .0036, year=2019, source=SCIEU_SOURCE),
    'warehouse': dict(label='Warehouse', intensity_kwh_m2=1.16 / .0036, year=2019, source=SCIEU_SOURCE),
    'hotel': dict(label='Hotel / motel', intensity_kwh_m2=1.28 / .0036, year=2019, source=SCIEU_SOURCE),
    'restaurant': dict(label='Restaurant', intensity_kwh_m2=2.58 / .0036, year=2019, source=SCIEU_SOURCE),
}
SCENARIO_MULTIPLIERS = {'low': .75, 'central': 1., 'high': 1.25}


def calculate_annual_heat(area_m2, building_type, options):
    """Convert site/final energy to useful space heat and DHW, never primary energy.

Efficiency factors are explicit scenario inputs: stock-average end-use shares do not
identify the installed equipment. A factor >1 represents an assumed seasonal COP.
"""
    kind = (building_type or '').lower()
    if kind not in ('residential', 'commercial'):
        raise ValueError(f'Canadian energy scenarios require residential/commercial types; found {building_type!r}. Use a delivered-intensity basis for other types.')
    residential = kind == 'residential'
    preset_id = 'residential' if residential else options['commercial_energy_preset']
    preset = ENERGY_PRESETS[preset_id]
    prefix = 'residential' if residential else 'commercial'
    intensity = options.get(prefix + '_total_intensity_kwh_m2')
    intensity = preset['intensity_kwh_m2'] if intensity is None else intensity
    multiplier = SCENARIO_MULTIPLIERS[options['energy_scenario']]
    total = area_m2 * intensity * multiplier
    space_fraction, water_fraction = options[prefix + '_space_fraction'], options[prefix + '_water_fraction']
    space_input, water_input = total * space_fraction, total * water_fraction
    space, water = space_input * options['space_heat_conversion'], water_input * options['water_heat_conversion']
    return dict(annual_kwh=space + water, annual_space_kwh=space, annual_dhw_kwh=water,
                total_final_energy_kwh=total, space_input_kwh=space_input, dhw_input_kwh=water_input,
                other_final_energy_kwh=total - space_input - water_input,
                energy_preset=preset_id, anchor_year=preset['year'], anchor_source=preset['source'],
                total_intensity_kwh_m2=intensity, intensity_overridden=options.get(prefix + '_total_intensity_kwh_m2') is not None,
                scenario_multiplier=multiplier, space_fraction=space_fraction, water_fraction=water_fraction,
                space_conversion=options['space_heat_conversion'], water_conversion=options['water_heat_conversion'],
                energy_source='Canadian benchmark scenario: final energy × end-use shares × assumed useful-heat conversion')
