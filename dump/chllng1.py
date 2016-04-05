from string import maketrans
text = "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj."

# abc = "abcdefghijklmnopqrstuvwxyzabc"

# textList = list(text)
# abcList = list(abc)

# result = ""
# print("size of text")
# print(len(textList))
# print("size of abc")
# print(len(abcList))

# for x in textList:
# 	if(x == " "):
# 		result+=x
# 	else:
# 		for y in abcList:
# 			if(x == y):
# 				index = abcList.index(y)
# 				result+=abcList[index+2]
# print(result)
abcd = "abcdefghijklmnopqrstuvwxyz"
abc= "cdefghijklmnopqrstuvwxyzab"
trans = maketrans(abcd,abc)

print(text.translate(trans))