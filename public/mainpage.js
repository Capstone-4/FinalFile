function mainpage() {
    return `<!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
        <title>우리집 지금 비었는데</title>
        <style>
        body {
          font-family: 'Maplestory';
          src: url('/Maplestory.ttf');
      }
      
      h4 {
          font-family: 'Maplestory';
          src: url('/Maplestory.ttf');
      }
      
      p {
          font-family: 'Maplestory';
          src: url('/Maplestory.ttf');
      }
      
      .container-fluid {
          width: 100vw;
          height: 100vh;
          background-color: #088bf7;
          align-items: center;
      }
      .title {
          width: 20vw;
          height: 20vh;
          color: white;
          float: left;
          font-size: 20px;
          margin-left: 80px;
          padding-top: 100px;
      }
      
      .collapse {
          width: 100vw;
          height: 30vh;
          margin-left: -50px;
          margin-top: 20px;
      }
      
      
      .nav-link {
          margin-left: 100px;
          color: white;
          text-align: center;
          font-size: 20px;
          width: 10vw;
          height: 10vh;
      }
      
      .btn_banner {
          width: 55vw;
          height: 40vh;
          float: right;
          padding-right: 30px;
          margin-left: 100px;
          float: right;
      
      }
      
      .image_btn{
          width: 15vw;
          height: 28vh;
          background-color: #0775e2;
          text-align: center;
          border-radius: 10px;
          margin-right: 20px;
          margin-top: 100px;
          float: right;
          color: white;
      }
      
      .house {
          width: 30vw;
          height: 40vh;
          float: left;
          margin-top: 100px;
          text-align: center;
          margin-left: 5%;
      }
      
      .back-img {
          width: 100%;
          height: 100%;
      }
      
      .text {
          height: 500px;
          margin-top: -350px;
          z-index: -1;
      }
      
      .btn_img {
          width: 100%;
          height: 80%;
      }
        </style>
    </head>
    <body>
    <div class="container-fluid">
        <div class="title">
            <img src="/public/home.png" width="30%" height="120%" style="float:left">
            <h4>empty house</h4>
            <p>우리집 지금 비었는데</p>
        </div>
          <div class="banner">
            <nav class="navbar navbar-expand-lg">
              <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavDropdown" aria-controls="navbarNavDropdown" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
              </button>
              <div class="collapse navbar-collapse" id="navbarNavDropdown">
                <ul class="navbar-nav">
                  <li class="nav-item">
                    <a class="nav-link" href="/researchhome">자리 찾기</a>
                  </li>
                  <li class="nav-item">
                    <a class="nav-link" href="/datapage">데이터 정보</a>
                  </li>
                  <li class="nav-item dropdown">
                    <a class="nav-link" href="#" role="button">
                      이용안내
                    </a>
                  </li>
                </ul>
              </div>
            </nav>
          </div>
          <div class="house">
              <img src="/배경.png" class="back-img">
                  <div>
                    <h4 class="text">당신의 자리를 찾아드립니다.</h4>
                  </div>
              </div>
              <div class="btn_banner">
                  <a class="image_btn" href="/search">
                    <h4>이용안내</h4>
                    <img src="/public/home.png" class="btn_img">
                  </a>
                  <a class="image_btn" href="/datapage">
                    <h4>데이터 조회</h4>
                    <img src="/home.png" class="btn_img">
                  </a>
                  <a class="image_btn" href="/researchhome" >
                    <h4>자리찾기</h4>
                    <img src="자리찾기.png" class="btn_img">
                  </a>
              </div>
          </div>
        </body>
    </html>`;
    }
    module.exports = mainpage;