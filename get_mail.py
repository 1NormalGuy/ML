mailTable=pd.DataFrame(columns=('Sender','Receiver','CarbonCopy','Subject','Date','Body','isSpam'))
f = open('trec06c/full/index', 'r')
csvfile=open('mailChinese.csv','w',newline='',encoding='utf-8')
writer=csv.writer(csvfile)
for line in f:
    str_list = line.split(" ")
    print(str_list[1])
 
    # 设置垃圾邮件的标签为0
    if str_list[0] == 'spam':
        label = '0'
    # 设置正常邮件标签为1
    elif str_list[0] == 'ham':
        label = '1'
    emlContent= emlAnayalyse('trec06c/full/' + str(str_list[1].split("\n")[0]))
    if emlContent is not None:
        writer.writerow([emlContent[0],emlContent[1],emlContent[2],emlContent[3],emlContent[4],emlContent[5],label])
