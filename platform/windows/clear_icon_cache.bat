@echo off
echo ========================================
echo Windows Icon Cache Clearer
echo ========================================
echo.
echo This will:
echo 1. Kill Windows Explorer
echo 2. Delete icon cache database
echo 3. Restart Explorer
echo.
pause

echo.
echo Killing Explorer...
taskkill /f /im explorer.exe

echo.
echo Deleting icon cache...
cd /d "%userprofile%\AppData\Local"
attrib -h IconCache.db
del IconCache.db /f /q

cd /d "%userprofile%\AppData\Local\Microsoft\Windows\Explorer"
del iconcache*.db /f /q 2>nul
del thumbcache*.db /f /q 2>nul

echo.
echo Restarting Explorer...
start explorer.exe

echo.
echo ========================================
echo Icon cache cleared!
echo ========================================
echo.
echo Please wait 10-15 seconds for Explorer to fully restart,
echo then check your icons again.
echo.
pause

