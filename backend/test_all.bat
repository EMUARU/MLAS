@echo off
REM ============================================================
REM  Multi-class FULL backend test - ALL models, ALL modes
REM  Run in backend directory: .\test_all.bat
REM ============================================================

echo ============================================================
echo  Step 0: Generate test data
echo ============================================================
python -c "from sklearn.datasets import make_classification; import pandas as pd; X,y=make_classification(n_samples=200,n_features=8,n_classes=2,random_state=42); df=pd.DataFrame(X,columns=[f'f{i}' for i in range(8)]); df['label']=y; df.to_csv('test_bin.csv',index=False); X,y=make_classification(n_samples=400,n_features=8,n_classes=4,n_informative=6,n_clusters_per_class=1,random_state=42); df=pd.DataFrame(X,columns=[f'f{i}' for i in range(8)]); df['label']=y; df.to_csv('test_mul.csv',index=False); pd.read_csv('test_bin.csv').drop(columns=['label']).to_csv('test_bin_nolabel.csv',index=False); pd.read_csv('test_mul.csv').drop(columns=['label']).to_csv('test_mul_nolabel.csv',index=False); print('OK')"

echo.
echo ============================================================
echo  Step 1: Train train_test_split (10 models x bin + mul = 20)
echo ============================================================

echo [1/20] XGB bin
python train_xgb.py test_bin.csv label train_test_split 0.8 m_xgb_bin 50 0.3 4 _t/tr_xgb_bin >nul 2>&1 && echo OK || echo FAIL
echo [2/20] XGB mul
python train_xgb.py test_mul.csv label train_test_split 0.8 m_xgb_mul 50 0.3 4 _t/tr_xgb_mul >nul 2>&1 && echo OK || echo FAIL
echo [3/20] LGBM bin
python train_lgbm.py test_bin.csv label train_test_split 0.8 m_lgbm_bin 100 0.1 -1 31 _t/tr_lgbm_bin >nul 2>&1 && echo OK || echo FAIL
echo [4/20] LGBM mul
python train_lgbm.py test_mul.csv label train_test_split 0.8 m_lgbm_mul 100 0.1 -1 31 _t/tr_lgbm_mul >nul 2>&1 && echo OK || echo FAIL
echo [5/20] RF bin
python train_rf.py test_bin.csv label train_test_split 0.8 m_rf_bin 100 50 0 -1 _t/tr_rf_bin >nul 2>&1 && echo OK || echo FAIL
echo [6/20] RF mul
python train_rf.py test_mul.csv label train_test_split 0.8 m_rf_mul 100 50 0 -1 _t/tr_rf_mul >nul 2>&1 && echo OK || echo FAIL
echo [7/20] LR bin
python train_lr.py test_bin.csv label train_test_split 0.8 m_lr_bin l2 0.0001 500 _t/tr_lr_bin >nul 2>&1 && echo OK || echo FAIL
echo [8/20] LR mul
python train_lr.py test_mul.csv label train_test_split 0.8 m_lr_mul l2 0.0001 500 _t/tr_lr_mul >nul 2>&1 && echo OK || echo FAIL
echo [9/20] SVM bin
python train_svm.py test_bin.csv label train_test_split 0.8 m_svm_bin 1.0 rbf _t/tr_svm_bin >nul 2>&1 && echo OK || echo FAIL
echo [10/20] SVM mul
python train_svm.py test_mul.csv label train_test_split 0.8 m_svm_mul 1.0 rbf _t/tr_svm_mul >nul 2>&1 && echo OK || echo FAIL
echo [11/20] MLP bin
python train_mlp.py test_bin.csv label train_test_split 0.8 m_mlp_bin 128 64 None relu 0.001 300 50 _t/tr_mlp_bin >nul 2>&1 && echo OK || echo FAIL
echo [12/20] MLP mul
python train_mlp.py test_mul.csv label train_test_split 0.8 m_mlp_mul 128 64 None relu 0.001 300 50 _t/tr_mlp_mul >nul 2>&1 && echo OK || echo FAIL
echo [13/20] AdaBoost bin
python train_adaboost.py test_bin.csv label train_test_split 0.8 m_ada_bin 100 1.0 3 _t/tr_ada_bin >nul 2>&1 && echo OK || echo FAIL
echo [14/20] AdaBoost mul
python train_adaboost.py test_mul.csv label train_test_split 0.8 m_ada_mul 100 1.0 3 _t/tr_ada_mul >nul 2>&1 && echo OK || echo FAIL
echo [15/20] CatBoost bin
python train_catboost.py test_bin.csv label train_test_split 0.8 m_cat_bin 100 0.1 4 _t/tr_cat_bin >nul 2>&1 && echo OK || echo FAIL
echo [16/20] CatBoost mul
python train_catboost.py test_mul.csv label train_test_split 0.8 m_cat_mul 100 0.1 4 _t/tr_cat_mul >nul 2>&1 && echo OK || echo FAIL
echo [17/20] TabNet bin
python train_tabnet.py test_bin.csv label train_test_split 0.8 m_tabnet_bin 256 5 10 _t/tr_tabnet_bin >nul 2>&1 && echo OK || echo FAIL
echo [18/20] TabNet mul
python train_tabnet.py test_mul.csv label train_test_split 0.8 m_tabnet_mul 256 5 10 _t/tr_tabnet_mul >nul 2>&1 && echo OK || echo FAIL
echo [19/20] TabPFN bin
python train_tabpfn.py test_bin.csv label train_test_split 0.8 m_tabpfn_bin _t/tr_tabpfn_bin >nul 2>&1 && echo OK || echo FAIL
echo [20/20] TabPFN mul
python train_tabpfn.py test_mul.csv label train_test_split 0.8 m_tabpfn_mul _t/tr_tabpfn_mul >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 2: Train k_fold (10 models x mul = 10)
echo ============================================================

echo [1/10] XGB kf bin
python train_xgb.py test_bin.csv label k_fold 3 m_xgb_kf 30 0.3 3 _t/kf_xgb_bin >nul 2>&1 && echo OK || echo FAIL
echo [2/10] XGB kf mul
python train_xgb.py test_mul.csv label k_fold 3 m_xgb_kf 30 0.3 3 _t/kf_xgb_mul >nul 2>&1 && echo OK || echo FAIL
echo [3/10] LGBM kf mul
python train_lgbm.py test_mul.csv label k_fold 3 m_lgbm_kf 50 0.1 -1 31 _t/kf_lgbm_mul >nul 2>&1 && echo OK || echo FAIL
echo [4/10] RF kf mul
python train_rf.py test_mul.csv label k_fold 3 m_rf_kf 50 30 0 -1 _t/kf_rf_mul >nul 2>&1 && echo OK || echo FAIL
echo [5/10] LR kf mul
python train_lr.py test_mul.csv label k_fold 3 m_lr_kf l2 0.0001 100 _t/kf_lr_mul >nul 2>&1 && echo OK || echo FAIL
echo [6/10] SVM kf mul
python train_svm.py test_mul.csv label k_fold 3 m_svm_kf 1.0 rbf _t/kf_svm_mul >nul 2>&1 && echo OK || echo FAIL
echo [7/10] MLP kf mul
python train_mlp.py test_mul.csv label k_fold 3 m_mlp_kf 128 64 None relu 0.001 100 20 _t/kf_mlp_mul >nul 2>&1 && echo OK || echo FAIL
echo [8/10] AdaBoost kf mul
python train_adaboost.py test_mul.csv label k_fold 3 m_ada_kf 50 1.0 3 _t/kf_ada_mul >nul 2>&1 && echo OK || echo FAIL
echo [9/10] CatBoost kf mul
python train_catboost.py test_mul.csv label k_fold 3 m_cat_kf 50 0.1 4 _t/kf_cat_mul >nul 2>&1 && echo OK || echo FAIL
echo [10/10] TabNet kf mul
python train_tabnet.py test_mul.csv label k_fold 3 m_tabnet_kf 256 3 5 _t/kf_tabnet_mul >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 3: Evaluate (10 models x mul = 10)
echo ============================================================

echo [1/10] eval XGB bin
python evaluate.py _t/tr_xgb_bin/m_xgb_bin.json test_bin.csv ev label pred _t/ev_xgb_bin >nul 2>&1 && echo OK || echo FAIL
echo [2/10] eval XGB mul
python evaluate.py _t/tr_xgb_mul/m_xgb_mul.json test_mul.csv ev label pred _t/ev_xgb_mul >nul 2>&1 && echo OK || echo FAIL
echo [3/10] eval LGBM mul
python evaluate.py _t/tr_lgbm_mul/m_lgbm_mul.pkl test_mul.csv ev label pred _t/ev_lgbm_mul >nul 2>&1 && echo OK || echo FAIL
echo [4/10] eval RF mul
python evaluate.py _t/tr_rf_mul/m_rf_mul.pkl test_mul.csv ev label pred _t/ev_rf_mul >nul 2>&1 && echo OK || echo FAIL
echo [5/10] eval LR mul
python evaluate.py _t/tr_lr_mul/m_lr_mul.pkl test_mul.csv ev label pred _t/ev_lr_mul >nul 2>&1 && echo OK || echo FAIL
echo [6/10] eval SVM mul
python evaluate.py _t/tr_svm_mul/m_svm_mul.pkl test_mul.csv ev label pred _t/ev_svm_mul >nul 2>&1 && echo OK || echo FAIL
echo [7/10] eval MLP mul
python evaluate.py _t/tr_mlp_mul/m_mlp_mul.pkl test_mul.csv ev label pred _t/ev_mlp_mul >nul 2>&1 && echo OK || echo FAIL
echo [8/10] eval AdaBoost mul
python evaluate.py _t/tr_ada_mul/m_ada_mul.pkl test_mul.csv ev label pred _t/ev_ada_mul >nul 2>&1 && echo OK || echo FAIL
echo [9/10] eval CatBoost mul
python evaluate.py _t/tr_cat_mul/m_cat_mul.pkl test_mul.csv ev label pred _t/ev_cat_mul >nul 2>&1 && echo OK || echo FAIL
echo [10/10] eval TabPFN mul
python evaluate.py _t/tr_tabpfn_mul/m_tabpfn_mul_tabpfn.pkl test_mul.csv ev label pred _t/ev_tabpfn_mul >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 4: Predict file (10 models x mul = 10)
echo ============================================================

echo [1/10] pred XGB bin
python predict.py _t/tr_xgb_bin/m_xgb_bin.json file _t/pr_xgb_bin --data_path test_bin_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [2/10] pred XGB mul
python predict.py _t/tr_xgb_mul/m_xgb_mul.json file _t/pr_xgb_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [3/10] pred LGBM mul
python predict.py _t/tr_lgbm_mul/m_lgbm_mul.pkl file _t/pr_lgbm_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [4/10] pred RF mul
python predict.py _t/tr_rf_mul/m_rf_mul.pkl file _t/pr_rf_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [5/10] pred LR mul
python predict.py _t/tr_lr_mul/m_lr_mul.pkl file _t/pr_lr_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [6/10] pred SVM mul
python predict.py _t/tr_svm_mul/m_svm_mul.pkl file _t/pr_svm_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [7/10] pred MLP mul
python predict.py _t/tr_mlp_mul/m_mlp_mul.pkl file _t/pr_mlp_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [8/10] pred AdaBoost mul
python predict.py _t/tr_ada_mul/m_ada_mul.pkl file _t/pr_ada_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [9/10] pred CatBoost mul
python predict.py _t/tr_cat_mul/m_cat_mul.pkl file _t/pr_cat_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL
echo [10/10] pred TabPFN mul
python predict.py _t/tr_tabpfn_mul/m_tabpfn_mul_tabpfn.pkl file _t/pr_tabpfn_mul --data_path test_mul_nolabel.csv --output_name out --label_column pred >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 5: Predict input (5 models x mul)
echo ============================================================

echo [1/5] input XGB mul
python predict.py _t/tr_xgb_mul/m_xgb_mul.json input _t/pi_xgb_mul --input_values 0.5 -1.2 0.3 0.8 -0.5 1.1 0.2 -0.7 >nul 2>&1 && echo OK || echo FAIL
echo [2/5] input LGBM mul
python predict.py _t/tr_lgbm_mul/m_lgbm_mul.pkl input _t/pi_lgbm_mul --input_values 0.5 -1.2 0.3 0.8 -0.5 1.1 0.2 -0.7 >nul 2>&1 && echo OK || echo FAIL
echo [3/5] input RF mul
python predict.py _t/tr_rf_mul/m_rf_mul.pkl input _t/pi_rf_mul --input_values 0.5 -1.2 0.3 0.8 -0.5 1.1 0.2 -0.7 >nul 2>&1 && echo OK || echo FAIL
echo [4/5] input CatBoost mul
python predict.py _t/tr_cat_mul/m_cat_mul.pkl input _t/pi_cat_mul --input_values 0.5 -1.2 0.3 0.8 -0.5 1.1 0.2 -0.7 >nul 2>&1 && echo OK || echo FAIL
echo [5/5] input MLP mul
python predict.py _t/tr_mlp_mul/m_mlp_mul.pkl input _t/pi_mlp_mul --input_values 0.5 -1.2 0.3 0.8 -0.5 1.1 0.2 -0.7 >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 6: Stacking single base (bin + mul)
echo ============================================================

echo [1/3] stk xgb-only bin
python train_stacking.py test_bin.csv "[\"xgb\"]" label xgb _t/stk_xgb_bin >nul 2>&1 && echo OK || echo FAIL
echo [2/3] stk xgb-only mul
python train_stacking.py test_mul.csv "[\"xgb\"]" label xgb _t/stk_xgb_mul >nul 2>&1 && echo OK || echo FAIL
echo [3/3] stk lgbm-only mul
python train_stacking.py test_mul.csv "[\"lgbm\"]" label lgbm _t/stk_lgbm_mul >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 7: Stacking pairs (each model paired with xgb, mul)
echo ============================================================

echo [1/9] xgb+lgbm
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label xgb _t/stk2_xgblgbm >nul 2>&1 && echo OK || echo FAIL
echo [2/9] xgb+rf
python train_stacking.py test_mul.csv "[\"xgb\",\"rf\"]" label xgb _t/stk2_xgbrf >nul 2>&1 && echo OK || echo FAIL
echo [3/9] xgb+lr
python train_stacking.py test_mul.csv "[\"xgb\",\"lr\"]" label xgb _t/stk2_xgblr >nul 2>&1 && echo OK || echo FAIL
echo [4/9] xgb+svm
python train_stacking.py test_mul.csv "[\"xgb\",\"svm\"]" label xgb _t/stk2_xgbsvm >nul 2>&1 && echo OK || echo FAIL
echo [5/9] xgb+mlp
python train_stacking.py test_mul.csv "[\"xgb\",\"mlp\"]" label xgb _t/stk2_xgbmlp >nul 2>&1 && echo OK || echo FAIL
echo [6/9] xgb+adaboost
python train_stacking.py test_mul.csv "[\"xgb\",\"adaboost\"]" label xgb _t/stk2_xgbada >nul 2>&1 && echo OK || echo FAIL
echo [7/9] xgb+catboost
python train_stacking.py test_mul.csv "[\"xgb\",\"catboost\"]" label xgb _t/stk2_xgbcat >nul 2>&1 && echo OK || echo FAIL
echo [8/9] xgb+tabnet
python train_stacking.py test_mul.csv "[\"xgb\",\"tabnet\"]" label xgb _t/stk2_xgbtabnet >nul 2>&1 && echo OK || echo FAIL
echo [9/9] xgb+tabpfn
python train_stacking.py test_mul.csv "[\"xgb\",\"tabpfn\"]" label xgb _t/stk2_xgbtabpfn >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 8: Stacking pairs bin (confirm binary still works)
echo ============================================================

echo [1/3] xgb+lgbm bin
python train_stacking.py test_bin.csv "[\"xgb\",\"lgbm\"]" label xgb _t/stk2b_xgblgbm >nul 2>&1 && echo OK || echo FAIL
echo [2/3] xgb+rf bin
python train_stacking.py test_bin.csv "[\"xgb\",\"rf\"]" label xgb _t/stk2b_xgbrf >nul 2>&1 && echo OK || echo FAIL
echo [3/3] xgb+catboost bin
python train_stacking.py test_bin.csv "[\"xgb\",\"catboost\"]" label xgb _t/stk2b_xgbcat >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 9: Stacking 3+ models (mul)
echo ============================================================

echo [1/4] xgb+lgbm+rf
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\",\"rf\"]" label xgb _t/stk3_xlr >nul 2>&1 && echo OK || echo FAIL
echo [2/4] xgb+lgbm+rf+lr
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\",\"rf\",\"lr\"]" label lgbm _t/stk4_xlrl >nul 2>&1 && echo OK || echo FAIL
echo [3/4] xgb+lgbm+catboost
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\",\"catboost\"]" label xgb _t/stk3_xlc >nul 2>&1 && echo OK || echo FAIL
echo [4/4] xgb+lgbm+adaboost+mlp
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\",\"adaboost\",\"mlp\"]" label xgb _t/stk4_xlam >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  Step 10: Stacking different meta models (mul)
echo ============================================================

echo [1/7] meta=xgb
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label xgb _t/stkm_xgb >nul 2>&1 && echo OK || echo FAIL
echo [2/7] meta=lgbm
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label lgbm _t/stkm_lgbm >nul 2>&1 && echo OK || echo FAIL
echo [3/7] meta=rf
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label rf _t/stkm_rf >nul 2>&1 && echo OK || echo FAIL
echo [4/7] meta=lr
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label lr _t/stkm_lr >nul 2>&1 && echo OK || echo FAIL
echo [5/7] meta=svm
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label svm _t/stkm_svm >nul 2>&1 && echo OK || echo FAIL
echo [6/7] meta=catboost
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label catboost _t/stkm_cat >nul 2>&1 && echo OK || echo FAIL
echo [7/7] meta=adaboost
python train_stacking.py test_mul.csv "[\"xgb\",\"lgbm\"]" label adaboost _t/stkm_ada >nul 2>&1 && echo OK || echo FAIL

echo.
echo ============================================================
echo  ALL TESTS COMPLETED
echo ============================================================