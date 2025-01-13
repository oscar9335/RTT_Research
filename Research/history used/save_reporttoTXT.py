# 生成 classification report 並保存
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(report_save_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {report_save_path}")