git config pull.rebase true
git pull origin master
rsync -avu --delete "static/images/" "assets/images"
git add .
git commit -m "msg"
git push origin master
