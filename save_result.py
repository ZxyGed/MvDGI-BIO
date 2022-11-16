import sqlite3
import pandas as pd


class ResultDB:
    def __init__(self, save_dir, dataset_name, table_name):
        self.conn = sqlite3.connect(f"{save_dir}/{name}.db")
        self.cur = self.conn.cursor()
        # sql column name is case-insensitive
        create_table = f'''
        create table if not exists {table_name}
        (moduleID char(32),
        classifierID char(32),

        module_params text,
        classifier_params text,

        acc_mean decimal(4,4),
        acc_std decimal(4,4),
        f1_mean decimal(4,4),
        f1_std decimal(4,4),
        microAupr_mean decimal(4,4),
        microAupr_std decimal(4,4),
        macroAupr_mean decimal(4,4),
        macroAupr_std decimal(4,4),
        zeroOneLoss_mean decimal(4,4),
        zeroOneLoss_std decimal(4,4));
        '''
        self.conn.execute(create_table)
        self.conn.commit()

    def __del__(self):
        self.conn.close()

    def search_ret(self, table_name, moduleID, classifierID):
        search_cmd = f'''
    	select * from {table_name} where moduleID={moduleID} and classifierID={classifierID};
    	'''
        self.cursor.execute(search_cmd)
        return self.cursor.fetchall()

    def insert_ret(self, table_name, moduleID, classifierID, module_params, classifier_params, ret_str, is_replace=False):
        ret = self.search_ret(table_name, moduleID, classifierID)
        if not len(ret) or is_replace:
            insert_cmd = f'''
    		insert into {table_name} values ({moduleID},{classifierID},{module_params},{classifier_params},{ret_str})
    		'''
            self.conn.execute(insert_cmd)
            self.conn.commit()

    def export2excel(self, table_name, save_dir):
        ret = self.search_ret(table_name, moduleID, classifierID)
        df = pd.DataFrame(ret)
        df.columns = ['moduleID', 'classifierID', 'module_params', 'classifier_params', 'acc_mean', 'acc_std', 'f1_mean',
                      'f1_std', 'microAupr_mean', 'microAupr_std', 'macroAupr_mean', 'macroAupr_std', 'zeroOneLoss_mean', 'zeroOneLoss_std']
        df.to_excel(f"{save_dir}/{table_name}.xlsx", index=False)
