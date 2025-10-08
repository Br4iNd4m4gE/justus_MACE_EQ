export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools
export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/martin_train.py  \
    --name="QEqWwater" \
    --train_file="water100.xyz" \
    --batch_size=10 \
    --valid_batch_size=5 \
    --valid_fraction=0.05 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1: -12.6746244439181, 8: -2041.03979050724}' \
    --model="maceQEq" \
    --hidden_irreps='64x0e+64x1o' \
    --r_max=8.0 \
    --max_num_epochs=50 \
    --device=cpu \
    --loss="charges_energy_forces" \
    --formal_charges_from_data \
    --charges_key="aims_charges" \
    --error_table="EFQRMSE" \
    --scale_atsize=1.0 \
    --charges_weight=100 \
    --restart_latest \
    >> output 
