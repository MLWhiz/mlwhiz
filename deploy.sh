rsync -avu --delete "static/images/" "assets/images"
hugo --minify
cd public
echo "mlwhiz.com" >CNAME
git add .
git commit -m "msg"
git push origin master
