from service.snowball_service import SnowballService

snowball_service  = SnowballService()


if __name__ == '__main__':
    print(snowball_service.suggest_stock('tyzn'))