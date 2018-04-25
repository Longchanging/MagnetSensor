@echo off  
title 批量替换文件名   
echo.  
echo 本批处理可批量替换本文件所在文件夹下所有文件的文件名。  
echo.  
echo.&set /p strtemp3= 请输入要添加前缀的文件类型：  
echo.&set /p strtemp2= 请输入要添加前缀的文件名字符串（不变则直接回车）：  
setlocal enabledelayedexpansion  
for /f "delims=" %%a in ('dir /a /b *.%strtemp3%') do (  
ren "%%~a" "%strtemp2%_%%a")  
echo.  
echo OK了！  
echo.  
pause  
