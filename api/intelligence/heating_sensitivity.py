"""Read-only factorial/OAT district-heating pipe-decision sensitivity study."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import math
import numpy as np
from rest_framework import serializers
from .heating_studies import HeatingStudyRequest
from .heating_models import MODEL_IDS, MODEL_CATALOG, generate_building_profile, NECB_SOURCE
from .heating_energy import calculate_annual_heat, ENERGY_SOURCES
from .heating_weather import fetch_canadian_weather
from .heating_topology import demonstration_case, snapshot_saved_network
from api.utils.pipe_sizing import RuntimeDesignCriteria, HydraulicFormulas

ASHRAE_SOURCE = 'https://handbook.ashrae.org/Handbooks/F21/SI/f21_ch18/f21_ch18_si.aspx'
DESIGN_SOURCE = 'https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/CAN_Canada/ON_Ontario/CAN_ON_Hamilton.Intl.AP.712630_TMYx.2011-2025.zip'
DESIGN_CONDITION = dict(station='Hamilton International Airport, WMO 712630', edition=2025,
    percentile='99.6%', outdoor_c=-18.4, source=DESIGN_SOURCE,
    note='Regional proxy for Burlington, not a Burlington station. ASHRAE conditions transcribed from the public DDY file; annual weather uses ERA5-Land, not TMY.')


class SensitivityRequest(serializers.Serializer):
    project_id = serializers.IntegerField(min_value=1)
    case = serializers.ChoiceField(choices=['demo','network'], default='demo')
    network_id = serializers.IntegerField(min_value=1, required=False, allow_null=True)
    connection_id = serializers.IntegerField(min_value=1, required=False, allow_null=True)
    years = serializers.ListField(child=serializers.IntegerField(min_value=1950,max_value=datetime.now(timezone.utc).year-1), min_length=1,max_length=10, default=lambda:list(range(datetime.now(timezone.utc).year-5,datetime.now(timezone.utc).year)))
    models = serializers.ListField(child=serializers.ChoiceField(choices=sorted(MODEL_IDS)),min_length=1,max_length=4,default=lambda:['hdd_daily_ca_v1','heat_balance_ca_v1','necb_slp_ca_v1'])
    resolutions = serializers.ListField(child=serializers.ChoiceField(choices=[1,3,6,24]), min_length=1,max_length=4,default=lambda:[1,3,6,24])
    compositions = serializers.ListField(child=serializers.ChoiceField(choices=['mixed','residential','nonresidential']),min_length=1,max_length=3,default=lambda:['mixed','residential','nonresidential'])
    experiments = serializers.ListField(child=serializers.ChoiceField(choices=['shape','input']),min_length=1,max_length=2,default=lambda:['shape','input'])
    energy_low = serializers.FloatField(min_value=.1,max_value=1,default=.75)
    energy_high = serializers.FloatField(min_value=1,max_value=3,default=1.25)
    envelope_low = serializers.FloatField(min_value=.05,max_value=10,default=.6)
    envelope_high = serializers.FloatField(min_value=.05,max_value=10,default=1.)
    ach_low = serializers.FloatField(min_value=0,max_value=10,default=.3)
    ach_high = serializers.FloatField(min_value=0,max_value=10,default=.7)
    gains_low = serializers.FloatField(min_value=0,max_value=100,default=2.)
    gains_high = serializers.FloatField(min_value=0,max_value=100,default=6.)
    range_justification = serializers.CharField(max_length=3000,default='Exploratory bounds: annual energy ±25%; envelope 0.6–1.0 W/m²K, effective ACH 0.3–0.7, gains 2–6 W/m². Not confidence intervals or code limits.')
    common = serializers.DictField(default=dict)
    indoor_design_c = serializers.FloatField(min_value=15,max_value=25,default=21.)
    design_percentile = serializers.ChoiceField(choices=['99','99.6'],default='99.6')
    include_dhw_design = serializers.BooleanField(default=True)
    available_pressure_pa = serializers.FloatField(min_value=1000,max_value=5000000,default=200000.)
    terminal_pressure_pa = serializers.FloatField(min_value=0,max_value=1000000,default=50000.)

    def validate(self, attrs):
        if any(isinstance(v,float) and not math.isfinite(v) for v in attrs.values()):
            raise serializers.ValidationError('All inputs must be finite.')
        for name in ('years','models','resolutions','compositions','experiments'):
            attrs[name]=list(dict.fromkeys(attrs[name]))
        if attrs['case']=='network' and not attrs.get('network_id'):
            raise serializers.ValidationError('Choose a saved network or the illustrative district.')
        if 'hdd_daily_ca_v1' not in attrs['models']:
            raise serializers.ValidationError('Include HDD as the transparent comparison baseline.')
        raw={**attrs['common'],'project_id':attrs['project_id'],'year':attrs['years'][0], 'models':attrs['models'],
             'reference':'hdd_daily_ca_v1','energy_basis':'canadian_scenario','energy_scenario':'central','normalize_physical':True}
        serializer=HeatingStudyRequest(data=raw);serializer.is_valid(raise_exception=True)
        attrs['common']=dict(serializer.validated_data)
        if attrs['common']['location']!='burlington':
            raise serializers.ValidationError('This initial study uses Burlington with a documented Hamilton regional design benchmark.')
        for low,high,key in [('envelope_low','envelope_high','envelope_w_m2k'),('ach_low','ach_high','air_changes_per_hour'),('gains_low','gains_high','internal_gains_w_m2')]:
            if not attrs[low] <= attrs['common'][key] <= attrs[high]:
                raise serializers.ValidationError(f'{low}/{high} must bracket the central {key}.')
        if attrs['terminal_pressure_pa'] >= attrs['available_pressure_pa']:
            raise serializers.ValidationError('Available pressure must exceed terminal pressure allowance.')
        return attrs


def prepare_sensitivity_case(options):
    return demonstration_case() if options['case']=='demo' else snapshot_saved_network(options['project_id'],options['network_id'],options.get('connection_id'))


def build_experiment_cases(options):
    cases=[]
    def add(experiment,label,model,factor=1.,physical=None):
        cases.append(dict(experiment=experiment,label=label,model=model,energy_factor=factor,physical=physical or {}, normalize_physical=physical is None))
    if 'shape' in options['experiments']:
        for model in options['models']: add('shape','Equal annual heat',model)
    if 'input' in options['experiments']:
        for name,factor in [('Low annual heat',options['energy_low']),('Central annual heat',1.),('High annual heat',options['energy_high'])]:
            for model in options['models']: add('input',name,model,factor)
        if 'heat_balance_ca_v1' in options['models']:
            add('input','Physical energy: central','heat_balance_ca_v1',physical={})
            cases[-1]['normalize_physical']=False
            for label,key,lo,hi in [('Envelope','envelope_w_m2k','envelope_low','envelope_high'),('Air exchange','air_changes_per_hour','ach_low','ach_high'),('Internal gains','internal_gains_w_m2','gains_low','gains_high')]:
                for side,value in [('low',options[lo]),('high',options[hi])]:
                    add('input',f'Physical energy: {label} {side}','heat_balance_ca_v1',physical={key:value})
    return cases


def hydraulic_arrays(load_kw, pipe, criteria, delta_t):
    """Vectorized existing laminar/transition/Swamee–Jain and Darcy–Weisbach rules."""
    load=np.asarray(load_kw,dtype=float)
    diameter=float(pipe['di'])
    velocity=load/(criteria.CP*delta_t*criteria.RHO*math.pi*diameter**2/4)
    reynolds=criteria.RHO*velocity*diameter/criteria.MU
    friction=np.zeros_like(velocity)
    laminar=(reynolds>0)&(reynolds<2000);trans=(reynolds>=2000)&(reynolds<4000);turb=reynolds>=4000
    friction[laminar]=64/reynolds[laminar];friction[trans]=.04
    friction[turb]=.25/np.log10(float(pipe.get('roughness',.000045))/(3.7*diameter)+5.74/reynolds[turb]**.9)**2
    return velocity,friction*criteria.RHO*velocity**2/(2*diameter)


def validate_criteria(criteria):
    for key in ('CP','RHO','MU','MAX_VELOCITY','TARGET_PRESSURE_DROP'):
        if not math.isfinite(getattr(criteria,key)) or getattr(criteria,key)<=0:
            raise ValueError(f'Invalid hydraulic constant {key}.')
    catalog=sorted(deepcopy(criteria.STD_PIPES),key=lambda p:float(p['di']))
    if not catalog: raise ValueError('Pipe catalogue is empty.')
    for p in catalog:
        for key in ('di','dn'):
            if not math.isfinite(float(p[key])) or float(p[key])<=0: raise ValueError('Invalid pipe dimensions.')
        if not math.isfinite(float(p.get('roughness',.000045))) or float(p.get('roughness',.000045))<0:
            raise ValueError('Invalid pipe roughness.')
        if p.get('cost') is not None and (not math.isfinite(float(p['cost'])) or float(p['cost'])<0):
            raise ValueError('Invalid catalogue cost.')
    if len({p['dn'] for p in catalog})!=len(catalog): raise ValueError('Pipe catalogue DN labels must be unique.')
    return catalog


def select_branch_pipe(peak, catalog, criteria, delta_t):
    if peak<=0: return None
    for p in catalog:
        v,g=hydraulic_arrays(np.array([peak]),p,criteria,delta_t)
        if v[0]<=criteria.MAX_VELOCITY and g[0]<=criteria.TARGET_PRESSURE_DROP: return p
    return None


def aggregate_branch_profiles(matrix, buildings, branches, composition):
    indexes={b['id']:i for i,b in enumerate(buildings)}
    eligible={b['id'] for b in buildings if composition=='mixed' or (b['building_type']=='residential')==(composition=='residential')}
    return np.array([matrix[[indexes[i] for i in branch['building_ids'] if i in eligible]].sum(axis=0) if any(i in eligible for i in branch['building_ids']) else np.zeros(matrix.shape[1]) for branch in branches])


def calculate_design_benchmark(buildings, branches, common, options, composition):
    """Simplified ASHRAE-method sensible space loss; separate optional hourly DHW allowance."""
    outdoor=-18.4 if options['design_percentile']=='99.6' else -15.6
    loads=[]
    for b in buildings:
        loss=(common['envelope_w_m2k']*b['floor_area']+.33*common['air_changes_per_hour']*b['volume_m3'])/1000
        loads.append([loss*max(options['indoor_design_c']-outdoor,0)])
    return aggregate_branch_profiles(np.array(loads),buildings,branches,composition)[:,0]


def summarize_stability(scenarios, branches):
    summaries=[]
    for composition in sorted({s['composition'] for s in scenarios}):
        for experiment in sorted({s['experiment'] for s in scenarios}):
            selected=[s for s in scenarios if s['composition']==composition and s['experiment']==experiment]
            for index,branch in enumerate(branches):
                values=[s['branches'][index] for s in selected if s['branches'][index]['peak_kw']>0]
                if not values: continue
                counts=Counter('no_fit' if r['dn'] is None else str(r['dn']) for r in values)
                valid=[r['dn'] for r in values if r['dn'] is not None]
                modal,count=counts.most_common(1)[0]
                summaries.append(dict(branch_id=branch['id'],name=branch['name'],composition=composition,experiment=experiment,
                    frequencies=dict(counts),scenario_count=len(values),modal_dn=None if modal=='no_fit' else float(modal),
                    changed_from_modal_pct=100*(1-count/len(values)),min_dn=min(valid) if valid else None,max_dn=max(valid) if valid else None,
                    stable=len(counts)==1 and 'no_fit' not in counts,no_fit_count=counts.get('no_fit',0)))
    return summaries


def run_pipe_sensitivity(options, case=None, progress=lambda message:None, weather_loader=fetch_canadian_weather):
    case=case or prepare_sensitivity_case(options)
    buildings=case['buildings'];branches=case['branches'];common=options['common']
    if not buildings: raise ValueError('The case has no consumer buildings.')
    criteria=RuntimeDesignCriteria(case.get('network_id'))
    catalog=validate_criteria(criteria);delta_t=common['delta_t_c']
    definitions=build_experiment_cases(options)
    count=len(definitions)*len(options['years'])*len(options['resolutions'])*len(options['compositions'])*len(branches)
    if count>120000: raise ValueError('Study exceeds 120,000 branch decisions. Reduce years, compositions or experiment scope.')
    inputs=[{**b,**calculate_annual_heat(b['floor_area'],b['building_type'],common)} for b in buildings]
    compositions=[c for c in options['compositions'] if c=='mixed' or any((b['building_type']=='residential')==(c=='residential') for b in buildings)]
    scenarios=[];weather=[];tests={c:[] for c in compositions};benchmark={};dhw_peaks={c:np.zeros(len(branches)) for c in compositions}
    for year in options['years']:
        progress(f'Retrieving and validating {year} weather')
        times,temperatures,provenance=weather_loader(year,common['location'])
        hdd=float(np.maximum(common['balance_temperature_c']-np.asarray(temperatures).reshape(-1,24).mean(axis=1),0).sum())
        weather.append({**provenance,'hdd':hdd,'minimum_hourly_c':float(np.min(temperatures))})
        envelopes={c:np.zeros((len(branches),len(times))) for c in compositions}
        shape_cache={}
        for definition in definitions:
            progress(f'{year}: {definition["label"]} · {definition["model"]}')
            trial={**common,**definition['physical'],'normalize_physical':definition['normalize_physical']}
            scaled=[{**b,**{k:b[k]*definition['energy_factor'] for k in ('annual_kwh','annual_space_kwh','annual_dhw_kwh')}} for b in inputs]
            matrix=np.array([generate_building_profile(definition['model'],temperatures,times,b,trial,shape_cache) for b in scaled])
            for composition in compositions:
                loads=aggregate_branch_profiles(matrix,inputs,branches,composition)
                envelopes[composition]=np.maximum(envelopes[composition],loads)
                for resolution in options['resolutions']:
                    averaged=loads.reshape(len(branches),-1,resolution).mean(axis=2)
                    decisions=[]
                    for i,branch in enumerate(branches):
                        peak=float(averaged[i].max());pipe=select_branch_pipe(peak,catalog,criteria,delta_t)
                        v,g=hydraulic_arrays(np.array([peak]),pipe,criteria,delta_t) if pipe else (np.array([0.]),np.array([0.]))
                        decisions.append(dict(branch_id=branch['id'],peak_kw=peak,annual_kwh=float(loads[i].sum()),
                            flow_kg_s=peak/(criteria.CP*delta_t),dn=pipe['dn'] if pipe else None,
                            status='inactive' if peak==0 else 'sized' if pipe else 'no_fit',
                            velocity_m_s=float(v[0]) if pipe else None,gradient_pa_m=float(g[0]) if pipe else None,
                            pipe_cost_cad=2*branch['length_m']*float(pipe['cost']) if pipe and pipe.get('cost') is not None else 0. if peak==0 else None))
                    costs=[d['pipe_cost_cad'] for d in decisions]
                    scenarios.append(dict(id=len(scenarios)+1,year=year,composition=composition,experiment=definition['experiment'],
                        model=definition['model'],label=definition['label'],energy_factor=definition['energy_factor'],physical=definition['physical'],
                        normalized=definition['normalize_physical'],resolution_hours=resolution,branches=decisions,
                        pipe_cost_cad=sum(costs) if all(c is not None for c in costs) else None))
        # DHW design allowance uses CENTRAL energy and the common hourly schedule, not normalized design space load.
        water_inputs=[{**b,'annual_space_kwh':0.,'annual_kwh':b['annual_dhw_kwh']} for b in inputs]
        water=np.array([generate_building_profile('hdd_daily_ca_v1',temperatures,times,b,common) for b in water_inputs])
        for composition in compositions:
            tests[composition].append(envelopes[composition])
            dhw_peaks[composition]=np.maximum(dhw_peaks[composition],aggregate_branch_profiles(water,inputs,branches,composition).max(axis=1))
    progress('Cross-checking selected diameters against common hourly scenarios and the design benchmark')
    by_dn={p['dn']:p for p in catalog};checks={}
    for composition in compositions:
        space=calculate_design_benchmark(inputs,branches,common,options,composition)
        water=dhw_peaks[composition] if options['include_dhw_design'] else np.zeros(len(branches))
        benchmark[composition]=[dict(branch_id=b['id'],space_kw=float(space[i]),dhw_allowance_kw=float(water[i]),total_kw=float(space[i]+water[i]),
                                    flow_kg_s=float(space[i]+water[i])/(criteria.CP*delta_t)) for i,b in enumerate(branches)]
        for i,branch in enumerate(branches):
            for dn,pipe in by_dn.items():
                hours=0;max_v=0.;max_g=0.
                for envelope in tests[composition]:
                    v,g=hydraulic_arrays(envelope[i],pipe,criteria,delta_t)
                    hours+=int(np.count_nonzero((v>criteria.MAX_VELOCITY)|(g>criteria.TARGET_PRESSURE_DROP)))
                    max_v=max(max_v,float(v.max()));max_g=max(max_g,float(g.max()))
                dv,dg=hydraulic_arrays(np.array([space[i]+water[i]]),pipe,criteria,delta_t)
                checks[(composition,i,dn)]=dict(common_violation_hours=hours,common_max_velocity_m_s=max_v,common_max_gradient_pa_m=max_g,
                    design_pass=bool(dv[0]<=criteria.MAX_VELOCITY and dg[0]<=criteria.TARGET_PRESSURE_DROP),design_gradient_pa_m=float(dg[0]))
    for scenario in scenarios:
        path_loss={case['root']:0.};design_path={case['root']:0.}
        all_pass=True
        for i,(branch,decision) in enumerate(zip(branches,scenario['branches'])):
            if decision['status']=='inactive':
                required_hours=sum(int(np.count_nonzero(e[i]>0)) for e in tests[scenario['composition']])
                design_required=benchmark[scenario['composition']][i]['total_kw']>0
                decision.update(common_violation_hours=required_hours,design_pass=not design_required,
                                common_max_gradient_pa_m=0. if required_hours==0 else None,
                                design_gradient_pa_m=0. if not design_required else None)
                all_pass &= required_hours==0 and not design_required
            elif decision['dn'] is None:
                decision.update(common_violation_hours=None,design_pass=False,common_max_gradient_pa_m=None,design_gradient_pa_m=None)
                all_pass=False
            else:
                decision.update(checks[(scenario['composition'],i,decision['dn'])])
                all_pass &= decision['common_violation_hours']==0 and decision['design_pass']
            gradient=decision['common_max_gradient_pa_m'];design_gradient=decision['design_gradient_pa_m']
            parent=path_loss.get(branch['start']);parent_design=design_path.get(branch['start'])
            path_loss[branch['end']]=parent+2*branch['length_m']*gradient if parent is not None and gradient is not None else None
            design_path[branch['end']]=parent_design+2*branch['length_m']*design_gradient if parent_design is not None and design_gradient is not None else None
        bound=max(path_loss.values()) if all(v is not None for v in path_loss.values()) else None
        design_bound=max(design_path.values()) if all(v is not None for v in design_path.values()) else None
        scenario['common_path_pressure_upper_bound_pa']=bound
        scenario['design_path_pressure_upper_bound_pa']=design_bound
        budget=options['available_pressure_pa']-options['terminal_pressure_pa']
        scenario['common_test_feasible']=bool(all_pass and bound is not None and design_bound is not None and max(bound,design_bound)<=budget)
    baselines={(s['year'],s['composition'],s['resolution_hours'],s['experiment'],s['label']):s
               for s in scenarios if s['model']=='hdd_daily_ca_v1'}
    for scenario in scenarios:
        key=(scenario['year'],scenario['composition'],scenario['resolution_hours'],scenario['experiment'])
        baseline=baselines.get((*key,scenario['label'])) or baselines.get((*key,'Central annual heat'))
        scenario['hdd_baseline_scenario_id']=baseline['id'] if baseline else None
        for i,decision in enumerate(scenario['branches']):
            base=baseline['branches'][i] if baseline else None
            decision['peak_difference_from_hdd_pct']=100*(decision['peak_kw']-base['peak_kw'])/base['peak_kw'] if base and base['peak_kw'] else None
            decision['dn_differs_from_hdd']=decision['dn']!=base['dn'] if base and decision['status']=='sized' and base['status']=='sized' else None
    stability=summarize_stability(scenarios,branches)
    cold=max(weather,key=lambda w:w['hdd'])['year']
    return dict(title='Sensitivity of district-heating pipe sizing to load-profile generation methods under limited demand data',
        options=options,case=case,building_inputs=inputs,scenarios=scenarios,stability=stability,weather=weather,coldest_tested_year=cold,
        skipped_compositions=[c for c in options['compositions'] if c not in compositions],
        benchmark=dict(condition={**DESIGN_CONDITION,'percentile':options['design_percentile']+'%','outdoor_c':-18.4 if options['design_percentile']=='99.6' else -15.6},
                       method_source=ASHRAE_SOURCE,indoor_c=options['indoor_design_c'],by_composition=benchmark),
        design=dict(delta_t_c=delta_t,cp_kj_kgk=criteria.CP,rho_kg_m3=criteria.RHO,mu_pa_s=criteria.MU,max_velocity_m_s=criteria.MAX_VELOCITY,
                    max_gradient_pa_m=criteria.TARGET_PRESSURE_DROP,catalog=catalog,available_pressure_pa=options['available_pressure_pa'],terminal_pressure_pa=options['terminal_pressure_pa']),
        sources=ENERGY_SOURCES+[NECB_SOURCE,ASHRAE_SOURCE,DESIGN_SOURCE,'https://open-meteo.com/en/docs/historical-weather-api'],
        assumptions=[options['range_justification'],
            'Shape runs normalize each building to identical annual space heat and DHW. Input-energy runs vary the common annual target; independent physical runs vary one parameter at a time and calculate space heat without normalization.',
            'Composition subsets retain the same nodes/edges; excluded buildings have zero demand. Missing categories are skipped, never fabricated. No random diversity factor is imposed.',
            'Hourly building demands are summed before branch peaks are taken. 3/6/24-hour values are non-overlapping means aligned to January 1 midnight in fixed local standard time.',
            'Each selected diameter is tested against the pointwise maximum of all generated hourly scenarios in the same composition and weather year. Violation hours count each tested hour once; they are not failure probabilities.',
            'Pressure upper bound sums each branch maximum over a supply/return path; maxima need not coincide. Exceeding this conservative bound calls for detailed hydraulic checking, not a claim of observed failure.',
            'Design benchmark uses sensible transmission plus effective ventilation loss, no gains, at 21 C or the selected indoor temperature. Floor-area envelope coefficient is a UA proxy, not a room-by-room ASHRAE calculation. It shares assumptions with heat balance and is not empirical validation.',
            'Optional design DHW is a separate central-scenario hourly allowance, not an ASHRAE space-heating load or instantaneous hot-water sizing calculation.',
            'Energy anchors use historical national averages and mixed source years. ±25% and physical ranges are exploratory until supported by local evidence. Weather changes shape while normalized annual totals are fixed.',
            'Cost is 2 × branch length × catalogue CAD/m for supply and return, with no trenches, fittings, pumps, installation or shared-route consolidation. Default catalogue diameters/costs are simplified research inputs.',
            'Feasibility is limited to maximum velocity, friction-gradient and pressure-budget screens; no pump curve, dispatch, thermal loss, loop balancing, storage or terminal-control simulation.',
            'Stability is reported separately by experiment and composition. Frequency is over enumerated scenarios, not a probability distribution; all-no-fit is never labelled stable.',
            'The coldest year is the highest HDD among tested years, not necessarily an extreme year in the long-term climate record.'])
