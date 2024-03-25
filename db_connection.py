import pymysql as sql


class Error(Exception):
    def __init__(self, message="", hint=""):
        """
        :param message:
        :param hint:
        """
        self.message = message
        self.hint = hint
        super().__init__(self.message)

    def __str__(self):
        """
        :return: string representation of the Error
        """
        return "Error message: " + self.message + "\n" + \
            "Hint: " + self.hint


# TODO: shared_memory 없애기

class DBConnection:
    def __init__(self, user: str, password: str, database: str):
        """
        :param user:
        :param password:
        :param database:
        """
        self.user = user
        self.database = database
        self.password = password
        self.host = 'localhost'
        self.charset = 'utf8'
        self.conn = sql.connect(host=self.host,
                                user=self.user,
                                password=self.password,
                                database=self.database,
                                charset=self.charset)

    def __str__(self):
        """
        :return: string representation of the DBConnection
        """
        return f"host: {self.host}\n" \
               f"user: {self.user}\n" \
               f"password: {self.password}\n" \
               f"database: {self.database}\n" \
               f"charset: {self.charset}"

    def __del__(self):
        """
        :explain: delete DBConnection
        """
        self.conn.close()

    def select(self, table: str, condition=None, order=None) -> tuple:
        """
        select(table='test')\n
        select(table='test', order={'id1': 1})\n
        select(table='test', condition=["id1", "id2"])\n
        select(table='test', condition=["id1"], order=[('id1', '=', 2)]\n
        :param table:
        :param condition: default = []
        :param order:  default = [()]
        :return: tuple
        :raises Error: pymysql exceptions
        """
        if order is None:
            order = [()]
        if condition is None:
            condition = []
        cursor = self.conn.cursor()
        res = None
        if len(condition) == 0 and len(order) == 0:
            sql_ = f"select * from {table}"
            try:
                cursor.execute(sql_)
            except Exception as e:
                raise Error(e.__str__(), "select error")
            res = cursor.fetchall()
        elif len(condition) == 0 and len(order) != 0:
            sql_ = f"select * from {table} where"
            for condition, value in order.items():
                if type("v") == type(value):
                    sql_ += f" {condition} = '{value}' AND"
                else:
                    sql_ += f" {condition} = {value} AND"
            sql_ = sql_.strip(" AND")
            try:
                cursor.execute(sql_)
            except Exception as e:
                raise Error(e.__str__(), "select error")
            res = cursor.fetchall()
        elif len(condition) != 0 and len(order) != 0:
            sql_ = "select "
            for c in condition:
                sql_ += f"{c}, "
            sql_ = sql_.rstrip(", ") + f" from {table} where"
            for c, o, v in order:
                if type("v") == type(v):
                    sql_ += f" {c}{o}'{v}' AND"
                else:
                    sql_ += f" {c}{o}{v} AND"
            sql_ = sql_.rstrip(" AND")
            try:
                cursor.execute(sql_)
            except Exception as e:
                raise Error(e.__str__(), "select error")
            res = cursor.fetchall()
        elif len(condition) != 0 and len(order) == 0:
            sql_ = "select "
            for c in condition:
                sql_ += f"{c}, "
            sql_ = sql_.rstrip(", ") + f" from {table}"
            try:
                cursor.execute(sql_)
            except Exception as e:
                raise Error(e.__str__(), "select error")
            res = cursor.fetchall()
        self.conn.commit()
        return res

    def insert(self, table: str, condition=None, values=None) -> bool:
        """
        insert(table="test", condition=["id1", "id2"], values=[[1, 2]])\n
        insert(table="test", condition=["id1", "id2"], values=[[1, 1], [2, 1], [3, 2], [4, 2]])
        :param table:
        :param condition: default = []
        :param values: default = [[]]
        :return: bool
        :raises Error: if condition is null or values is null,
        if condition and values are not same length, pymysql exceptions
        """
        if values is None:
            values = [[]]
        if condition is None:
            condition = []
        cursor = self.conn.cursor()
        if len(condition) == 0 and len(values) == 0:
            raise Error(message="condition == Null and values == Null", hint="condition, values")
        elif len(condition) == 0 and len(values) != 0:
            raise Error(message="condition == Null", hint="condition")
        elif len(condition) != 0 and len(values) == 0:
            raise Error(message="values == Null", hint="values")
        else:
            for i in range(len(values)):
                if len(condition) != len(values[i]):
                    raise Error(message="len(condition) != len(values)", hint=f"values[{i}]")
            sql_ = f"insert into {table} ("
            for x in condition:
                sql_ += f"{x}, "
            sql_ = sql_.rstrip(", ") + ") values"
            for i in range(len(values)):
                sql_ += "("
                for x in values[i]:
                    if type("x") == type(x):
                        sql_ += f"'{x}', "
                    else:
                        sql_ += f"{x}, "
                sql_ = sql_.rstrip(", ") + "), "
            sql_ = sql_.rstrip(", ")
            try:
                cursor.execute(sql_)
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise Error(e.__str__(), "insert error")

            return True

    def delete(self, table: str, order=None) -> bool:
        """
        delete(table="test", order="[("id", "=", 1), ("id2", "=", "test"))
        :param table:
        :param order: default = {}
        :return: bool
        :raises Error: if order is null, pymysql exceptions
        """
        if order is None:
            order = [()]
        cursor = self.conn.cursor()
        if len(order) == 0:
            raise Error(message="order == Null", hint="order")
        else:
            sql_ = f"delete from {table} where"
            for c, o, v in order:
                if type("v") == type(v):
                    sql_ += f" {c}{o}'{v}' AND"
                else:
                    sql_ += f" {c}{o}{v} AND"
            sql_ = sql_.rstrip(" AND")
        try:
            cursor.execute(sql_)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Error(e.__str__(), "delete error")
        return True

    def update(self, table: str, condition=None, values=None, order=None) -> bool:
        """
        update(table="test", condition=["id1", "id2"], values=["val1", 1], order=[("id1", "=", "id"), ...])
        :param table:
        :param condition: default = []
        :param values: default = []
        :param order: default = [()]
        :return: bool
        :raises Error: if condition or values or order is null, pymysql exceptions
        """
        if order is None:
            order = [()]
        if values is None:
            values = []
        if condition is None:
            condition = []
        cursor = self.conn.cursor()
        err = [False, False, False]
        errFlag = False
        if len(condition) == 0:
            err[0] = "condition"
        if len(order) == 0:
            err[1] = "order"
        if len(values) == 0:
            err[2] = "values"
        msg = ""
        for x in err:
            if x:
                errFlag = True
                msg += x + ", "
        msg = msg.rstrip(", ")
        if errFlag:
            raise Error(message=msg + "is empty", hint=msg)
        else:
            if len(condition) != len(values):
                raise Error(message="len(condition) != len(values)", hint=f"values")
            sql_ = f"update {table} set "
            for c, v in zip(condition, values):
                if type("v") == type(v):
                    sql_ += f"{c}='{v}', "
                else:
                    sql_ += f"{c}={v}, "
            sql_ = sql_.rstrip(", ") + " where"
            for c, o, v in order:
                if type("v") == type(v):
                    sql_ += f" {c}{o}'{v}' AND"
                else:
                    sql_ += f" {c}{o}{v} AND"
            sql_ = sql_.rstrip(" AND")
            try:
                cursor.execute(sql_)
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise Error(e.__str__(), "update error")
            return True
