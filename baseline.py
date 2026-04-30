import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print(f"학습 데이터 크기: {train.shape}")
print(f"테스트 데이터 크기: {test.shape}")


TARGET = 'avg_delay_minutes_next_30m'
ID_COLS = ['ID', 'layout_id', 'scenario_id']

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
print(f"피처 수: {len(feature_cols)}")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    print(f"── Fold {fold + 1} ──")
    X_tr = train.loc[tr_idx, feature_cols]
    y_tr = train.loc[tr_idx, TARGET]
    X_val = train.loc[val_idx, feature_cols]
    y_val = train.loc[val_idx, TARGET]

    model = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=7,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test[feature_cols]) / 5


oof_mae = mean_absolute_error(train[TARGET], oof_preds)
print(f"OOF MAE: {oof_mae:.4f}")


submission = pd.DataFrame({'ID': test['ID'], TARGET: test_preds})
submission.to_csv('./submission.csv', index=False)
print("submission.csv 저장 완료.")
