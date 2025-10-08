export PYTHONPATH=${PYTHONPATH}:/work/mace_qeq_dev/mace-tools
export PYTHONPATH=${PYTHONPATH}:/work/mace_qeq_dev/graph_longrange

python saveModel.py --name="NaClWaterWill" --train_file="water_will.xyz" --batch_size=40 --valid_fraction=0.05 --config_type_weights='{"Default":1.0}' --E0s='{1: -12.6746244439181, 8: -2041.03979050724}' --model="FullQEqMace" --hidden_irreps='64x0e' --r_max=8.0 --max_num_epochs=2000 --device=cuda --loss="charges_energy_forces" --formal_charges_from_data --charges_key="aims_charges" --error_table="MartinCharges" --scale_atsize=1.0 --restart_latest >> output 
