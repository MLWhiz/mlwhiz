rsync -avu --delete "static/images/" "assets/images"
hugo --minify
cd public
git pull origin master
echo "mlwhiz.com" >CNAME
git add .
git commit -m "msg"
git push origin master
cd ..
git pull origin master
git add .
git commit -m "msg"
git push origin master
