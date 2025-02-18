#!/usr/bin/env bash

set -x

mkdir -p plots

Inputs=Inputs

# for proc in NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70 NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100 NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70 NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100 NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70 NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100 NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_50 NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_70 NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_100 NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_50 NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_70 NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_100; do
#  python plotting/plot_input_features.py --input ${Inputs}/low_mass_trigger.parquet --summary ${Inputs}/low_mass_trigger.json --sig-proc ${proc} -o plots/low_mass_trigger_${proc}
#  python plotting/plot_input_features.py --input ${Inputs}/low_mass_trigger.parquet --summary ${Inputs}/low_mass_trigger.json --sig-proc ${proc} -o plots/low_mass_trigger_${proc}_norm --norm
# done

# for proc in NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70 NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100 NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70 NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100 NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70 NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100 NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_50 NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_70 NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_100 NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_50 NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_70 NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_100; do
#  python plotting/plot_input_features.py --input ${Inputs}/low_mass_trigger_2018.parquet --summary ${Inputs}/low_mass_trigger_2018.json --sig-proc ${proc} -o plots/low_mass_trigger_2017_${proc}
#  python plotting/plot_input_features.py --input ${Inputs}/low_mass_trigger_2018.parquet --summary ${Inputs}/low_mass_trigger_2018.json --sig-proc ${proc} -o plots/low_mass_trigger_2017_${proc}_norm --norm
# done

# for proc in radionM500_HHggTauTau radionM800_HHggTauTau radionM1000_HHggTauTau; do
#   python plotting/plot_input_features.py --input ${Inputs}/Pass3/low_mass_trigger_2017.parquet --summary ${Inputs}/Pass3/low_mass_trigger_2017.json --sig-proc ${proc} -o plots/low_mass_trigger_2017_trigger_applied_${proc}
#   python plotting/plot_input_features.py --input ${Inputs}/Pass3/low_mass_trigger_2017.parquet --summary ${Inputs}/Pass3/low_mass_trigger_2017.json --sig-proc ${proc} -o plots/low_mass_trigger_2017_trigger_applied_${proc}_norm --norm
# done

# for proc in XToHHggTauTau_M1000 XToHHggTauTau_M250 XToHHggTauTau_M260 XToHHggTauTau_M270 XToHHggTauTau_M280 XToHHggTauTau_M290 XToHHggTauTau_M300 XToHHggTauTau_M320 XToHHggTauTau_M350 XToHHggTauTau_M400 XToHHggTauTau_M450 XToHHggTauTau_M500 XToHHggTauTau_M550 XToHHggTauTau_M600 XToHHggTauTau_M650 XToHHggTauTau_M700 XToHHggTauTau_M750 XToHHggTauTau_M800 XToHHggTauTau_M900; do
#   python plotting/plot_input_features.py --input ${Inputs}/Graviton.parquet --summary ${Inputs}/summary.json --sig-proc ${proc} -o plots/Graviton_${proc}
#   python plotting/plot_input_features.py --input ${Inputs}/Graviton.parquet --summary ${Inputs}/summary.json --sig-proc ${proc} -o plots/Graviton_${proc}_norm --norm
# done

for proc in XToHHggTauTau_M260 ; do
  python plotting/plot_input_features.py --input ${Inputs}/merged_nominal.parquet --summary ${Inputs}/summary.json --sig-proc ${proc} -o plots/Graviton_${proc}
  #python plotting/plot_input_features.py --input ${Inputs}/Graviton.parquet --summary ${Inputs}/summary.json --sig-proc ${proc} -o plots/Graviton_${proc}_norm --norm
done

#python plotting/plot_input_features.py --input ${Inputs}/merged_nominal.parquet --summary ${Inputs}/summary.json --sig-proc XToHHggTauTau_M300 XToHHggTauTau_M500 XToHHggTauTau_M1000 -o plots/Graviton_mix2
