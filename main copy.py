from konlpy.tag import Okt

okt = Okt()

sentence='저는 이백이십일만원짜리 컴퓨터를 맞추고 싶고 영상 편집과 리그오브레전드를 하고 싶습니다.'

print(okt.nouns(sentence))
print(okt.morphs(sentence))
print(okt.phrases(sentence))
print(okt.pos(sentence))
