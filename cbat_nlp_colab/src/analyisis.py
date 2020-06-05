id2label = {'0': '過去',
 '1': '過去-最近',
 '2': '最近（1か月以内）',
 '3': '現在（状態、性質、考えなど）',
 '4': '過去-現在（習慣など）',
 '5': '未来（予定、予測、願望、仮定など）',
 '6': '現在-未来',
 '7': '最近-未来',
 '8': '過去-未来',
 '9': '最近-現在（習慣など）'}

# write file for error analysis
with open("./analysis/test.txt") as f_gold, open("./analysis/test_results_original_epoch10.txt") as f_pred, open("./analysis/error_analysis_epoch10.txt", 'w') as f_out:
    f_pred.readline()
    f_out.write("original\tgold_id\tpred_id\n")
    for g_line, p_line in zip(f_gold, f_pred):
        g_line = g_line.strip()
        sent, gold_id = g_line.split('\t')
        p_line = p_line.strip()
        _, pred_id = p_line.split('\t')

        f_out.write(sent + '\t' + id2label[gold_id] + '\t' + id2label[pred_id] + '\n')