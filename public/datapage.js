function datapage() {
    return `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
        <link href="/datapage.css" rel="stylesheet">
        <title>DataPage</title>
        <style>
        .container-fluid {
            width: 100vw;
            height: 1200px;
        }

        h4,li{
            font-family: 'Maplestory';
            src: url('/Maplestory.ttf');
        }
        
        a{
            color: white;
            text-decoration-line: none;
            font-family: 'Maplestory';
            src: url('/Maplestory.ttf');
            margin-left: 12%;
        }

        .nav-link1 {
            color: white;
            text-align: center;
            font-size: 20px;
            margin-left: 0%;
        }
        
        .title {
            width: 20vw;
            height: 20vh;
            color: white;
            float: left;
            font-size: 20px;
            margin-top: 2%;
            margin-left: 5%;
        }
        .top-banner {
            width: 101.5%;
            height: 9.5%;
            background-color: #088bf7;
        }
        
        .collapse {
            width: 100%;
            height: 30vh;
            margin-left: -10%;
            margin-top: -4%;
        }
        .nav-link {
            margin-left: 10vw;
            margin-bottom: 11%;
            color: white;
            text-align: center;
          font-size: 20px;
        }
        
        .local-anl {
            width: 10%;
            height: 5%;
            color: black;
            margin-left: 8.5%;
            margin-top: 2%;
        }
        
        .btn{
            width: 20%;
            border-color: black;
        }
        
        .btn-banner{
            text-align: center;
            margin-top: 3%;
            width: 100%;
            margin-left: -12px;
            height: 15vh;
        }
        
        .btn-banner2 {
            margin-top: 0.5%;
        
        }
        
        .tp{
            width: 25%;
            height: 8%;
            float: left;
            margin-left: 8.5%;
            margin-top: 50px;
        }
        
        
        .date {
            width: 25%;
            height: 20%;
            float: left;
        }
        
        
        .form-group {
            width: 100%;
        }
        
        .work {
            width: 300px;
        }
        
        .confirm-ban {
            width: 100%;
            margin-left: -12px;
            text-align: center;
            clear: both;
            height: 3%;
        }
        
        .btn1 {
            width: 7%;
        }
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="title">
                <h4>empty house</h4>
                <a class="nav-link1" href="/mainpage">우리집 지금 비었는데</a>
            </div>
            <div class="top-banner">
                <div class="banner">
                    <nav class="navbar navbar-expand-lg">
                        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavDropdown" aria-controls="navbarNavDropdown" aria-expanded="false" aria-label="Toggle navigation">
                            <span class="navbar-toggler-icon"></span>
                        </button>
                        <div class="collapse navbar-collapse" id="navbarNavDropdown">
                            <ul class="navbar-nav">
                            <li class="nav-item">
                            <a class="nav-link" href="researchhome">자리 찾기</a>
                            </li>
                            <li class="nav-item">
                            <a class="nav-link" href="datapage">데이터 조회</a>
                            </li>
                            <li class="nav-item dropdown">
                            <a class="nav-link"  href="#">이용안내</a>
                            </li>
                            </ul>
                        </div>
                    </nav>
                </div>
            </div>
            <div class="local-anl">
                <h4 class="loc-anl" style="font-size: 18px;">조회 데이터 정보</h4>
            </div>
            <div class="btn-banner">
                <button type="button" class="btn btn-light" id="shop-num2">점포수</button>
                <button type="button" class="btn btn-light" id="life-time">신생기업 생존률</button>
                <button type="button" class="btn btn-light" id="avg-sales">평균영업기간</button>
                <button type="button" class="btn btn-light" id="type-sth">업종별 발달</button>
                <div class="btn-banner2">
                    <button type="button" class="btn btn-light" id="house-num">가구수</button>
                    <button type="button" class="btn btn-light" id="flow-peo">유동인구</button>
                </div>
            </div>
            <div class="tp">
                <h5>기준 분기</h5>
                <div class="date">
                    <form>
                        <div class="form-group col-md-4">
                            <select id="inputState" class="form-control">
                            <option selected>2022년</option>
                            <option>2023년</option>
                            </select>
                        </div>
                    </form>
                </div>
                <div class="date" style="margin-left: 1%;">
                    <form>
                        <div class="form-group col-md-4">
                            <select id="inputState" class="form-control">
                            <option selected>1분기</option>
                            <option>2분기</option>
                            <option>3분기</option>
                            <option>4분기</option>
                            </select>
                        </div>
                    </form>
                </div>
            </div>
            <div class="tp" style="margin-left: -80px;">
                <h5 id="title">생활 밀집 업종</h5>
                <div class="date">
                    <form>
                        <div class="form-group col-md-4">
                            <select id="inputState" class="form-control">
                            <option selected>전체</option>
                            <option>강남구</option>
                            <option>강동구</option>
                            <option>강북구</option>
                            <option>강서구</option>
                            </select>
                        </div>
                    </form>
                </div>
                <div class="date" style="margin-left: 1%;">
                    <form>
                        <div class="work form-group col-md-4">
                            <select id="inputState" class="form-control">
                            <option selected>전체</option>
                            <option>강남구</option>
                            <option>강동구</option>
                            <option>강북구</option>
                            <option>강서구</option>
                            </select>
                        </div>
                    </form>
                </div>
            </div>
            <div class="confirm-ban">
                <button type="button" class="btn1 btn btn-primary">적용</button>
            </div>
        </div>
    </body>
    <script src="/datapage.js"></script>
    </html> `;
    }
    module.exports = datapage;