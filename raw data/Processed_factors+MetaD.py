

f_original = open('./processed_factors_221112.txt', 'r')
f_attach = open('./MetaD221112.txt', 'r')

f = open('./total_result_221112.txt', 'w')

for i in range(13):
    line = f_original.readline()
    line_attach = f_attach.readline()

    # print(line[:-1])
    # print(line_attach[:-1])

    '''
    if i == 164:
        f.write('\n')
        continue
    '''
    
    '''
    f.write(line[:-1])
    f.write('\t')
    '''
    line_elements = line[:-1].split('\t')
    line_text = ''
    
    for e in line_elements:
        #print(e)
        if e == 'nan' or e == 'inf':
            line_text += '1e+8'
        elif e == '-inf':
            line_text += '-1e+8'
        else:
            if e[-1] == '\t':
                line_text += e[:-1]
            else:
                line_text += e

        line_text += '\t'

    f.write(line_text)
    line_elements_meta = line_attach[:-1].split('\t')
    line_text_meta = ''
    '''
    for e in line_elements_meta:
        line_text_meta += e
        line_text_meta += '\t'
    '''
    line_text_meta += line_elements_meta[0]
    line_text_meta += '\t'
    line_text_meta += line_elements_meta[1]
    line_text_meta += '\n'
    
    f.write(line_text_meta)
    # f.write('\n')

f_original.close()
f_attach.close()
f.close()
