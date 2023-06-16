function home(centerLatLng, markers) {
  if (!centerLatLng || !centerLatLng.lat) {
    // centerLatLng 객체가 유효하지 않거나 lat 속성이 없는 경우 기본값 설정
    centerLatLng = { lat: 37.5665, lng: 126.9780 };
  }
  return `<!DOCTYPE html>
  <html>
    <head>
      <meta charset="utf-8"/>
      <title>Kakao 지도 시작하기</title>
      <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey=58ade57e1d4d5f4677a833c1b7a69515&libraries=clusterer"></script>
      <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
      <style>
        #info{
          width: 60em;
          height: 100%;
          border: 2px solid lightgray;
        }
        .container-fluid {
            width: 100vw;
            height: 100vh;
            position: relative;
            z-index: 10;
        }
        
        a{
            color: white;
            text-decoration-line: none;
        }
        
        .title {
            width: 20vw;
            height: 20vh;
            color: white;
            float: left;
            font-size: 20px;
            margin-top: 18px;
            margin-left: 80px;
        }
        .top-banner {
            width: 100vw;
            height: 7vw;
            background-color: #088bf7;
        }
        
        .collapse {
            width: 100%;
            height: 30vh;
            margin-left: -100px;
            margin-top: -50px;
        }
        .nav-link {
            margin-left: 10vw;
            margin-bottom: 30px;
            color: white;
            text-align: center;
            font-size: 20px
        }
        
        .input-p{
            width: 23vw;
            height: 75vh;
            margin-left: 10px;
        }
        
        .input-title-banner{
            width: 100%;
            height: 100%;
            border-radius: 9px;
            background-color: white;
        
        }
        
        .btn {
            background-color: #088bf7;
        }
        
        .btn1 {
            width: 40%;
        }
        .btn2 {
            width: 40%;
            margin-left: 15px;
        }
        
        
        .input-title{
            color: #088bf7;
            margin-left:20px;
            font-size: 18px;
            margin-top: 50px;
        }
        
        .map {
            width: 100%;
            height: 120%;
            background-color: antiquewhite;
            position: absolute;
            z-index: -1;
        }
        
        .form-group{
            width: 80%;
        }
        
        .form-control {
            margin-left: 8%;
            box-sizing: content-box;
        }
        
        .button-banner{
            text-align: center;
            margin-top: 20px;
        
        }
        
        .four_btn {
            margin-left: 4%;
            margin-top: 3px;
            width: 20%;
            height: 160%;
            float: left;
        }
        
        .btn-light {
            width: 100%;
            height: 80%;
        }
        
        .search-banner {
            width: 100%;
            height: 30%;
            margin-top: 10px;
        }
        
        .b{
            width: 100%;
            height: 95%;
            background-color: white;
        }
        
        
        .four_btn_banner {
           margin-top: 20px;
            width: 100%;
            height: 10%;
        }
        
        .btn3{
            width: 45%;
            background-color: #088bf7;
            color: white;
        }
        
        h5 {
            margin-top: 5%;
            margin-left: 7%;
        }
        .qut-banner {
            width: 100%;
            height: 10%;
            text-align: center;
        }
        
        .btn5 {
            width: 18%;
            height: 100%;
            background-color: lightgray;
            border-color: lightgray;
            font-size: 7px;
        }
        
        .btn6 {
            font-size: 7px;
            background-color: lightgrey;
            border-color: lightgray;
        }
        
        .can-btn {
            background-color: lightgray;
            border-color: lightgray;
        }
        
        .under {
            margin-top: 50px;
        }
        
        .result-banner {
            width: 23vw;
            height: 40%;
            background-color: white;
            display: none;
            margin-top: 10px;
            border-radius: 9px;
        }
        
        .pho {
            width: 50%;
            height: 50%;
            background-color: violet;
        }
      </style>
    </head>
    <body>
    <div class="container-fluid">
    <div class="map">
    <div id="map" style="width: 100%; height: 100%;"></div>
    </div>
        <div class="title">
            <img src="아이콘.png" width="50px" style="float:left">
            <h4>empty house</h4>
            <a href="/mainpage">우리집 지금 비었는데</a>
        </div>
        <div class="top-banner">
            <div class="banner">
                <nav class="navbar navbar-expand-lg">
                <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavDropdown" aria-controls="navbarNavDropdown" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNavDropdown">
                    <ul class="navbar-nav">
                    <li class="nav-item active">
                        <a class="nav-link" href="/search">뜨는 상권</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/researchhome">자리 찾기</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/datapage">분석 데이터 정보</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#">이용안내</a>
                    </li>
                    </ul>
                </div>
            </nav>
            </div>
        </div>
        <div class="input-p">
            <div class="search-banner">
                <div class="input-title-banner">
                    <h4 class="input-title">뜨는 상권</h4>
                    
                    <form action="/search" method="get">
                        <div class="form-group col-md-4">
                            <select id="search-input" name="q" class="form-control">
                            <option selected>서울시 전체</option>
                            <option>강남구</option>
                            <option>강동구</option>
                            <option>강북구</option>
                            <option>강서구</option>
                            <option>관악구</option>
                            <option>광진구</option>
                            <option>구로구</option>
                            <option>금천구</option>
                            <option>노원구</option>
                            <option>도봉구</option>
                            <option>동대문구</option>
                            <option>동작구</option>
                            <option>마포구</option>
                            <option>서대문구</option>
                            <option>서초구</option>
                            <option>성동구</option>
                            <option>성북구</option>
                            <option>송파구</option>
                            <option>양천구</option>
                            <option>영등포구</option>
                            <option>용산구</option>
                            <option>은평구</option>
                            <option>종로구</option>
                            <option>중구</option>
                            <option>중랑구</option>
                            </select>
                        </div>
                    </form>
                    <div class="button-banner">
                        <button type="button" class="btn1 btn btn-primary" id="hot-vil">뜨는 동네</button>
                        <button type="button" class="btn2 btn btn-primary" id="hot-com">뜨는 상권</button>
                    </div>
                </div>
            </div>
            <div class="b" id="main_banner">
                <div class="four_btn_banner">
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="shop-num">점포수</button>
                    </div>
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="sales">매출</button>
                    </div>
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="flo-peo">유동인구</button>
                    </div>
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="live-peo">주거인구</button>
                    </div>
                </div>
                <div>
                    <h5>기준</h5>
                </div>
                <div style="text-align: center;">
                    <button type="button" class="btn3 btn btn-primary" id="rat-st">증가율 기준</button>
                    <button type="button" class="btn3 btn btn-primary" id="abs-st">절대값 기준</button>
                </div>
                <div>
                    <h5>기간</h5>
                </div>
                <div class="qut-banner">
                    <button type="button" class="btn5 btn btn-light" id="y-2022-1">2022년 1분기</button>
                    <button type="button" class="btn5 btn btn-light" id="y-2022-2">2022년 2분기</button>
                    <button type="button" class="btn5 btn btn-light" id="y-2022-3">2022년 3분기</button>
                    <button type="button" class="btn5 btn btn-light" id="y-2022-4">2022년 4분기</button>
                    <button type="button" class="btn5 btn btn-light" id="y-2023-1">2023년 1분기</button>
                </div>
                <div>
                    <h5>업종</h5>
                </div>
                <div class="four_btn_banner">
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="all">전체</button>
                    </div>
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="food-btn">요식업</button>
                    </div>
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="service-btn">서비스업</button>
                    </div>
                    <div class="four_btn">
                        <button type="button" class="btn6 btn btn-light" id="sell-btn">소매업</button>
                    </div>
                </div>
                <div class="under button-banner">
                    <button type="button" class="can-btn btn1 btn btn-primary" id="reset">초기화</button>
                    <button type="button" class="ad btn1 btn btn-primary" id="okay">적용</button>
                </div>
            </div>
            <div class="result-banner" id="result">
                <div>
                    <p>건물리스트</p>
                </div>
            </div>
        </div>
    </div>
</div>

      <p class="auto" id="info">information</p>
      <script>
        const container = document.getElementById('map');
        const options = {
          center: new window.kakao.maps.LatLng(${centerLatLng.lat}, ${centerLatLng.lng}),
          level: 3
        };

        const map = new kakao.maps.Map(container, options);
  
                              
        kakao.maps.load(() => { 
        
          const markers = ${JSON.stringify(markers)};

          function addMarkers() {
            const clusterer = new kakao.maps.MarkerClusterer({
              map: map,
              averageCenter: true,
              minLevel:3,
              
              texts: getTexts
            });

            const bounds = map.getBounds();
            const positions = markers.filter(function(position) {
              return bounds.contain(new kakao.maps.LatLng(position.latlng.lat, position.latlng.lng));
            });
          
            let visibleMarkersCount;
            const zoomLevel = map.getLevel();
            if (zoomLevel <= 3) {
              visibleMarkersCount = Math.round(markers.length * 0.1); // 수정된 부분
            } else if (zoomLevel <= 4) {
              visibleMarkersCount = Math.round(markers.length * 0.2); // 수정된 부분
            } else if (zoomLevel <= 5) {
              visibleMarkersCount = Math.round(markers.length * 0.3); // 수정된 부분
            } else if (zoomLevel <= 6) {
              visibleMarkersCount = Math.round(markers.length * 0.5); // 수정된 부분
            } else {
              visibleMarkersCount = Math.round(markers.length * 1); // 수정된 부분
            }
            function getTexts(count){
              if(zoomLevel <= 3) {
                return Math.round(count*10);        
              } else if(zoomLevel <= 4) {
                return Math.round(count*5);
              } else if(zoomLevel <= 5) {
                return Math.round(count*10/3);
              } else if(zoomLevel <=6) {
                return Math.round(count*2);
              } else{
                return count;
              }
            }
         
            const visibleMarkers = positions
              .sort(() => 0.5 - Math.random()) 
              .slice(0, visibleMarkersCount);
            
            
            const markersToAdd = visibleMarkers.map(function(position) {
              const marker = new kakao.maps.Marker({
                position: new kakao.maps.LatLng(position.latlng.lat, position.latlng.lng),
              });

              kakao.maps.event.addListener(marker, 'click', function() { // 수정된 부분
                const link = "https://m.land.naver.com/article/info/" + position.atclNo;
                const clickedMarkerCoords = new kakao.maps.LatLng(position.latlng.lat, position.latlng.lng);
                const markersWithSameCoords = markers.filter(function(marker) {
                  const markerCoords = new kakao.maps.LatLng(marker.latlng.lat, marker.latlng.lng);
                  return markerCoords.equals(clickedMarkerCoords);
              });
              const content = markersWithSameCoords.map(function(marker) {
                return '<p>' + marker.atclFetrDesc + ', ' + marker.atclNm +
                ', ' + marker.tradTpNm + ', ' + marker.flrInfo + ', ' +
                marker.prc + ', ' + marker.direction + ', ' +
                '<a href="' + link + '">' + link + '</a></p>';
                }).join('');
                info.innerHTML = content;
              });
  
              return marker;
            });
            
          
            clusterer.clear();
            clusterer.addMarkers(markersToAdd);

          }
  
          kakao.maps.event.addListener(map, 'idle', function() {
            addMarkers();
          });
      });
      </script>
    </body>
  </html> `;
  }
  module.exports = home;