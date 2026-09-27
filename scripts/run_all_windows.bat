@echo off
REM Full pipeline: indexes -> train -> predict -> 5 validated submissions.
REM Uses %PY% if set, else "py -3" (Python launcher), else "python".
cd /d "%~dp0\.." || (echo repo folder not found & pause & exit /b 1)
if not defined PY (
  where py >nul 2>&1 && (set "PY=py -3") || (set "PY=python")
)
echo Using: %PY%
echo ===== 0. packages and tests =====
%PY% -m pip install -r requirements.txt || goto :fail
%PY% -m pytest -q || goto :fail
echo ===== 1. blocking indexes (train + test) =====
%PY% -m src.preprocessing.build_indexes --config configs/pipeline.yaml || goto :fail
echo ===== 2. train + validate =====
%PY% -m src.train || goto :fail
echo ===== 3. score the test set =====
%PY% -m src.predict || goto :fail
echo ===== 4. five submissions (v1 = best, written last so output\ holds it) =====
for %%V in (2 3 4 5 1) do (
  %PY% -m src.make_submission --variant %%V || goto :fail
)
echo.
echo ALL DONE. output\matching_results.tsv + output\candidate_pairs.tsv = variant 1 (best on validation).
echo Other variants: data\interim\submissions\   Report: experiments\results\
pause
exit /b 0
:fail
echo.
echo A STEP FAILED - scroll up for the error. Finished steps are kept; rerun from the failed step.
pause
exit /b 1
