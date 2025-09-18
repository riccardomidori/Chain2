import pandas as pd
import pytz
import connectorx as cx
import pprint



class Utils:
    @staticmethod
    def get_data(house_id, connection_string, freq="15m"):
        k = freq.find("m")
        minutes = float(freq[:k]) * 60

        query = (f"SELECT min(unixtime) as unixtime, "
                 f"avg(a_act_power) as a , "
                 f"avg(b_act_power) as b, "
                 f"avg(c_act_power) as c "
                 f"FROM ned_em_{house_id} "
                 f"WHERE unixtime>=unix_timestamp(curdate()) - 30*3*86400 "
                 f"and unixtime < unix_timestamp(curdate()) "
                 f"GROUP BY floor(unixtime / %s)"
                 )
        df = cx.read_sql(connection_string, query % minutes, index_col="unixtime")
        df.index = pd.to_datetime(df.index, unit="s", utc=True).tz_convert(pytz.timezone("Europe/Rome"))

        return df


def auto_str(cls):
    def __str__(self):
        params = {
            k: v for k, v in vars(self).items()
        }
        return "%s\n%s" % (
            type(self).__name__,
            pprint.pformat(params, indent=2)
        )

    cls.__str__ = __str__
    return cls
