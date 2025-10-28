@echo off
echo ========================================
echo FIX GIT LARGE FILES - XOA KHOI HISTORY
echo ========================================
echo.

echo CANH BAO: Script nay se xoa cac file lon khoi Git history!
echo Backup du lieu truoc khi chay!
echo.
pause

echo.
echo Buoc 1: Xoa cache Git...
git rm -r --cached data/
git rm -r --cached results/

echo.
echo Buoc 2: Commit thay doi...
git add .gitignore
git commit -m "Remove large files from tracking"

echo.
echo Buoc 3: Xoa cac file lon khoi Git history...
echo (Co the mat vai phut...)

git filter-branch --force --index-filter "git rm --cached --ignore-unmatch data/processed/X_val.npz" --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch data/processed_improved/X_train.npz" --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch data/processed_lstm_method/balanced_X.npy" --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch results/models/1dcnn_best_original.h5" --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch results/models/1dcnn_best_weighted.h5" --prune-empty --tag-name-filter cat -- --all

echo.
echo Buoc 4: Xoa tat ca file trong data/ va results/...
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch data/" --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch results/" --prune-empty --tag-name-filter cat -- --all

echo.
echo Buoc 5: Cleanup...
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo.
echo ========================================
echo HOAN THANH!
echo ========================================
echo.
echo Kiem tra kich thuoc repository:
git count-objects -vH

echo.
echo Bay gio ban co the push:
echo   git push -f