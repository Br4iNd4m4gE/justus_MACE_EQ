python eval_configs.py --configs="water_will.xyz" --model="NaClWaterWill.model" --output="qeqCharges.xyz" --device=cuda
python eval_configs.py --configs="neg.xyz" --model="NaClWaterWill.model" --output="qeqChargesNeg.xyz" --device=cuda
python eval_configs.py --configs="pos.xyz" --model="NaClWaterWill.model" --output="qeqChargesPos.xyz" --device=cuda
python eval_configs.py --configs="neu.xyz" --model="NaClWaterWill.model" --output="qeqChargesNeut.xyz" --device=cuda

