## 行为检测系统 v1.0
### 项目简介
行为检测系统是一款基于 vue2 + webpack 框架开发的系统。
### baseUrl: http://localhost:8080
## 登录
### 1. 登录系统
- **URL**: `/login`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "username": "admin",
    "password": "123456"
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
        "code": 200,
        "data": {
          "token": "123456", // token
          "userInfo": {
            "id": 1,
            "username": "admin",
            "password": "123456",
            "roles": "admin", // 角色
            "permissions": ["*"], // 权限
          }
        },
        "message": "登录成功"
    }
  ```
### 2. 获取用户信息
- **URL**: `/userInfo`
- **Method**: `GET`
- **Request Body**:
  ```json
  {
    "id": "1" // 用户id
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
        "code": 200,
        "data": {
            "id": 1,
            "username": "admin",
            "password": "123456",
            "roles": "admin", // 角色
            "permissions": ["*"], // 权限
            "createTime": "2022-08-08 10:16:58" // 创建时间
        },
        "message": "成功"
    }
  ```
#### 3. 修改密码
- **URL**: `/change-password`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "id": "lgl123456",
    "password": "123456"
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
        "code": 200,
        "message": "密码修改成功"
    }
  ```
## 用户管理
### 1. 查询列表
- **URL**: `/user/infoList`
- **Method**: `GET`
- **Request Body**:
  ```json
  {

  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
        "code": 200,
        "data": [
          {
              "id": "1", // 用户id
              "username": "admin", //   用户名
              "password": "123456", // 
              "role": "超级管理员", // 角色
              "permissions": ["*"], // 权限
              "createTime": "2022-08-08 10:16:58", // 创建时间
          },{
              "id": "2", // 用户id
              "username": "user", //   用户名
              "password": "123456", // 
              "role": "超级管理员", // 角色
              "permissions": ["*"], // 权限
              "createTime": "2022-08-08 10:16:58", // 创建时间
          }
        ],
        "message": "成功"
    }
    ```
### 2. 新增、修改用户
- **URL**: `/user/addModifyInfo`
- **Method**: `POST | PUT`
- **Request Body**:
  ```json
  {
    "id": "1", // 修改需传
    "username": "admin", //   用户名
    "password": "123456", // 密码hash值
    "role": "超级管理员", // 角色
    "permissions": ["*"], // 权限
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
        "code": 200,
        "data": [
          {   
              "id": "1", // 用户id,修改时需传
              "username": "admin", //   用户名
              "password": "123456", // 密码hash值
              "role": "超级管理员", // 角色 
              "permissions": ["*"], // 权限
              "createTime": "2022-08-08 10:16:58", // 创建时间
          }
        ],
        "message": "成功"
    }
    ```
### 3. 删除用户
- **URL**: `/user/deleteUser`
- **Method**: `DELETE`
- **Request Body**:
  ```json
  {
    "id": "1", // 用户id
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
        "code": 200,
        "message": "成功"
    }
    ```
## 一、基本信息
### 1. 查询版本信息
- **URL**: `/basic/version`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
        "code": 200,
        "data": {
            BoardId: "RJ-MLU-560DE5C68A2DC4FAD38480CB60383608", // 设备标识(如何获取)？
            Version: "0.0.54",                                  // 软件版本(如何获取)？全部软件+1
            BoardTemp: "40 (C)",                                // 芯片温度(如何获取)？
            StorageSpace: "2.9G/total"                               // 存储空间(是userdata里面的空间吗，还是可用空间
            System: "4.5.4",                                    // 系统版本(如何获取)？/userdata/data/etc/version
            Time: "2025-07-17 14:09:38"                         // 盒子时间(是板卡时间吗)？
        }
        "message": "成功"
    }
    ```

### 3. 同步时间
- **URL**: `/basic/syncTime`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "cmd": "2025-08-08 10:16:58"
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 4. 重启服务
- **URL**: `/basic/app_system_kill`
- **Method**: `POST`
- **Request Body**:

  ```json
  {}
  ```

- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 5. 重启设备

- **URL**: `/basic/reboot`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "cmd": "reboot"
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

## 二、通道管理

### 1. 查询列表

- **URL**: `/tube/alg_media_fetch`
- **Method**: `GET`
- **Request Params**:
  ```json
  {
    "pageSize": 10, // 每页条数
    "pageNum": 1, // 当前页码
    "keyword": "" // 关键字
  }
  ```
- **Response**:

  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": 
        [{
          "id": 1,
          "MediaDesc": "",
          "MediaName": "1",	   		// 相机编号
          "MediaStatus": "2",			//设置的是0未知 1连接中 2正常 3连接失败 4视频流不存在 
          "MediaUrl": "rtsp://192.168.100.140:8550/1",
          "ProtocolType": 0, （界面不显示）
          "created_at": "2022-08-08 10:16:58"
        },
        {	
          "id": 2,
          "MediaDesc": "",
          "MediaName": "2",	   		// 相机编号
          "MediaStatus": "2",			//设置的是0未知 1连接中 2正常 3连接失败 4视频流不存在 
          "MediaUrl": "rtsp://192.168.100.141:8551/1",
          "ProtocolType": 0, （界面不显示）
          "created_at": "2022-08-08 10:16:58"
        }],
      "message": "成功",
      "totalItems": 2
    }
    ```

  ```

  ```

#### 2. 新增 | 修改
- **URL**: `/tube/alg_media_config`
- **Method**: `POST | PUT`
- **Request Body**:
  ```json
  {
    "MediaName": "1", 	// 相机编号
    "MediaUrl": "1", 		// 视频地址
    "MediaStatus": "1", // 通道状态，我设置的是0未知 1连接中 2正常 3连接失败 4视频流不存在 
    "MediaDesc": "1",   // 相机描述
    "ProtocolType": 0 （界面不显示）
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

#### 3. 删除
- **URL**: `/tube/alg_media_delete`
- **Method**: `DELETE`
- **Request Body**:
  ```json
  {
    "id": "1" 			// Tube id
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

## 三、任务管理

### 1. 查询列表

- **URL**: `/task/alg_task_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": 
        {
          "id": 4,
          "AlgTaskSession": "任务1",    		// 任务编号
          "ControlCommand": 1,     // 1: 启动 0: 停止
          "TaskDesc": "测试1",
          "AlgTaskStatus": "0",    // 设置的是 1连接中 2正常 3连接失败 4视频流不存在  5
          "AlarmBody": 0,          // warning list里面的id
          "AlarmProtocol": 0,
          "MediaName": "3",        // 相机编号
          "MediaUrl": "rtsp://192.168.9.140:8554/3",   // 视频源地址
          "ScheduleId": -1,     // 保留，先不做
          "UserData":     // UserData里面的value用字符串的形式发送
              [{
                    "name":"p2pnetconfig",    //算法名称
                    "baseAlgname":"人员拥挤检测",    // 算法配置信息
                    "enabled":true,    // 是否勾选启用该算法（人员拥挤检测）
                    "confThresh":0.5,    // 检测阈值（默认值0.5）
                    "normalRange":{"min":10,"max":20}    // 正常人数范围：[min, max]（判断spare / normal / crowded）
                },
                    {
                    "name":"p2pnetconfig2",    // 算法配置信息
                    "baseAlgname":"遗留物检测",     // 是否勾选启用该算法（人员拥挤检测）
                    "enabled":false,
                    "confThresh":0.5     // 检测阈值（默认值0.5）
              }],    
          "created_at": "2025-09-22 10:28:15"
        },
      "message": "成功"
    }
    ```

### 2. 新增 | 修改
- **URL**: `/task/alg_task_config`
- **Method**: `POST/PUT`
- **Request Body**:
  ```json
  {
    "AlarmBody":  0,
    "AlarmProtocol": 0,
    "AlgTaskSession":"11", // 任务编号
    "ControlCommand": 1, // 1: 启动 0: 停止
    "AlgTaskStatus":"2",	//设置的是 1连接中 2正常 3连接失败 4视频流不存在  5
    "MediaName": "1",        // 视频源
    "MetadataUrl": "",      // 上报地址
    "TaskDesc": "1",         // 任务描述
    "ScheduleId": -1,		//保留，先不做
    "UserData":[{	
             		"name":"p2pnetconfig",											//算法名称
             		"baseAlgname":"人员拥挤检测",            		// 算法配置信息
            		"enabled": true,                            // 是否勾选启用该算法（人员拥挤检测）
            		"confThresh": 0.5,                          // 检测阈值（默认值0.5）
            		"normalRange": { "min": 10, "max": 20 }    // 正常人数范围：[min, max]（判断spare / normal / crowded）
             	},
							{
                "name":"p2pnetconfig2",	
                "baseAlgname":"口罩检测",            				// 算法配置信息
                "enabled": true,                            // 是否勾选启用该算法（人员拥挤检测）
                "confThresh": 0.5                          // 检测阈值（默认值0.5）
              }]
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```
### 5. 停止 | 启动

- **URL**: `/task/alg_task_control`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "id": 1, // 任务编号
    "ControlCommand": 1 // 1: 启动 0: 停止
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 6. 删除(任务停止状态才可删除)

- **URL**: `/task/alg_task_delete`
- **Method**: `DELETE`
- **Request Body**:
  ```json
  {
    "id": 1 // 任务编号
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

## 四、 实时预览（先预留）

### 1. 获取通道树形结构

- **URL**: `/preview/app_preview_channel`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data":[
        {
          "name": "合流通道",
          "chn": [
            {
              "name": "通道1",
              "ptz": false,			//是啥
              "task": "",
              "url": "group/1"
            }
          ]
        },
        {
        	"name": "任务通道",
          "chn": [
            {
              "name": "11",
              "ptz": false,
              "task": "11",
              "url": "task/11"
            },
            {
              "name": "测试乘客密度1",
              "ptz": false,
              "task": "测试乘客密度",
              "url": "task/测试乘客密度"
            }
          ]
        }
      ],
      "message": "成功"
    }
    ```

## 五、 告警管理

### 1. 获取列表

- **URL**: `/warning/alg_alarm_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [{
                "id": 1,
                "MediaName": "1",                           //视频通道
                "MediaUrl": "rtsp://192.168.100.1:554/1",   //视频通道url
                "ResultType":["口罩"],                       //告警类型 
                "ResultDescription":"",     // 告警任务(详情里面是告警类别)
                "imgPath":"",                              //图片路径 
                "UploadReason": "", // 上报原因
                "Uploadstatus": "1", // 上报状态 1: 已上报 2: 未上报
                "Uploadvideo_path": "", // 上报视频路径
                "UserData": [{}, {}], // 算法涉及的信息
                "created_at": "2021-08-12 10:00:00"

               },
               {
								"id": 2,
								"MediaName": "2",                           //视频通道
                "MediaUrl": "rtsp://192.168.100.1:554/2",   //视频通道url
                "ResultType":["拥挤度-拥挤"],                //告警类型 
                "ResultDescription":"",     // 告警任务(详情里面是告警类别)
                "imgPath":"",                              //图片路径 
                "UploadReason": "",                        // 上报原因
                "Uploadstatus": "1",      // 上报状态 1: 已上报 2: 未上报
                "Uploadvideo_path": "",                    // 上报视频路径
                "UserData": [{}, {}],                       // 算法涉及的信息
                "created_at": "2021-08-12 10:00:00"
               }],
      "message": "成功"
    }
    ```
 2.新增告警
- **URL**: `/warning/alg_alarm_fetch`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "MediaName": "1",                           //视频通道
    "MediaUrl": "rtsp://192.168.100.1:554/1",   //视频通道url
    "ResultType":["口罩"],                       //告警类型 
    "ResultDescription":"",     // 告警任务(详情里面是告警类别)
    "imgPath":"",                              //图片路径 
    "UploadReason": "", // 上报原因
    "Uploadstatus": "1", // 上报状态 1: 已上报 2: 未上报
    "Uploadvideo_path": "", // 上报视频路径
    "UserData": [{}, {}] // 算法涉及的信息
    
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": {
                "id":5,
                "MediaName": "1",                           //视频通道
                "MediaUrl": "rtsp://192.168.100.1:554/1",   //视频通道url
                "ResultType":["口罩"],                       //告警类型 
                "ResultDescription":"",     // 告警任务(详情里面是告警类别)
                "imgPath":"",                              //图片路径 
                "UploadReason": "", // 上报原因
                "Uploadstatus": "1", // 上报状态 1: 已上报 2: 未上报
                "Uploadvideo_path": "", // 上报视频路径
                "UserData": [{}, {}], // 算法涉及的信息
                "created_at": "2021-08-12 10:00:00"
               },
      "message": "成功"
    }
    ```
### 2. 删除选中		//每小时删一次，保持48小时的图片
- **URL**: `/warning/alg_alarm_delete`
- **Method**: `DELETE`
- **Request Body**:
  ```json
  {
    "Where": [16727, 16728] 		// warning table 序号id，*id以及其路径里面的图片也要一起删除*
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

## 六、 网络配置

### 1. 获取列表

- **URL**: `/ethernet/app_network_query_v2`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "name": "eth0",                 // 网络名称
          "address": "255.255.255.0",     // 网络地址
          "mask": "192.168.100.1",        // 子网掩码
          "gateway": "114.114.114.114",   // 网关地址
          "mac": "68:af:ff:10:08:aa",     // 物理地址
        }
      ],
      "message": "成功"
    }
    ```

### 2. 编辑

- **URL**: `/ethernet/app_network_apply_v2`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "name": "eth0",
    "address": "192.168.100.61",
    "gateway": "192.168.100.1",
    "mask": "255.255.0.1",
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "已修改,正在重启设备"
    }
    ```

## 七、 参数配置(先不做)

### 1. 获取列表
//参数为什么是数组？Table数据要固定，一个key一个value
- **URL**: `/param/alg_config_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "key": "DimOfAlarmImage", // 参数标识
          "desc": "告警图片尺寸大小（0 宽高640x480,1原始相机尺寸，默认0）", // 参数描述
          "type": 0, // 类型
          "value": "0" // 参数值
        }
      ],
      "message": "成功"
    }
    ```

### 2. 添加 | 修改

- **URL**: `/param/alg_config_save`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "key": "测试",
    "desc": "",
    "value": "1",
    "type": 1,
    "editable": true,
    "index": -1
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 3. 删除

- **URL**: `/param/alg_config_delete`
- **Method**: `DELETE`
- **Request Body**:
  ```json
  {
    "key": "测试"
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 4. 导入
//导入的文件是什么格式的？
- **URL**: `/param/alg_config_import`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    file: (binary)
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

## 八、 计划模版（先不做）

### 1. 获取模版信息
//summary是数组用字符串方式交互可以吗？
- **URL**: `/schedule/alg_schedule_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "Id": -1,
          "Name": "默认模板", // 模版标识
          "Summary": "全部日期", // 工作计划
          "Value": ""
        }
      ],
      "message": "成功"
    }
    ```

### 2. 新增模版

- **URL**: `/schedule/alg_schedule_create`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "name": "测试", // 计划名称
    "summary": ["星期二 07:00~12:00", "星期三 07:00~12:00"], // 日期时间
    "value": "000000000000000000000000000000000000000000000000000000000000001111111111000000000000000000000000000000000000001111111111000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 3. 删除

- **URL**: `/schedule/alg_schedule_delete`
- **Method**: `DELETE`
- **Request Body**:
  ```json
  {
    "id": 5 // 模版ID
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "删除成功"
    }
    ```

## 九、其他配置

### 1. 获取模型信息列表

- **URL**: `/config/alg_plugin_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [{
								"desc": "识别人员抽烟", 	//  算法描述
								baseAlgname："抽烟" 			// 涉及算法
							},
							{
								"desc": "打电话", 				//  算法描述
								baseAlgname："打电话" 		// 涉及算法
							}],
      "message": "成功"
    }
    ```

### 3. 获取告警类别列表(先不做）

- **URL**: `/config/alg_alarm_type_list`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "class": "明烟明火检测", 		// 所属算法
          "desc": "检测到明火", 			// 告警描述
          "major": "8",
          "minor": "46",
          "permitted": false,
          "type": "Fire", 						// 告警类别
          "wav": "" 									// 语音文件
        }
      ],
      "message": "成功"
    }
    ```

### 4. 告警类别-上传语音
- **URL**: `/config/app_upload_voice`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    file: (binary)
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 5. 获取阈值配置列表

- **URL**: `/config/alg_threshold_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
		        {
		          "desc": "未戴安全帽阈值", // 参数描述
		          "id": 0,
		          "value": 0.699999988079071 // 参数值
		        }
		        ],
      "message": "成功"
    }
    ```

### 6. 阈值配置-参数编辑

- **URL**: `/config/alg_threshold_config`
- **Method**: `post`
- **Request Body**:
  ```json
  {
    "id": 0,
    "value": 0.71
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

## 十、轮询配置(先不做，先做前面的基础功能)

### 1. 相机管理-获取相机列表

- **URL**: `/camera/app_rtsp_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "CapUrl": "",
          "captureFile": "Content/BG/BG_5.jpg", // 底图预览路径
          "captureTime": "2025-06-10 10:02:55", // 底图时间
          "rtspId": 5,
          "rtspName": "测试超载", // 相机名称
          "rtspUrl": "rtsp://192.168.100.140:8554/3" // 地址
        }
      ],
      "message": "成功"
    }
    ```

### 2. 相机管理-添加 ONVIF 相机

- **URL**: `/camera/add_onvif`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "OnvifUser": "测试001", // 用户名
    "OnvifPassword": "123456", // 密码
    "DeviceIp": "0.0.0.0", // 网络地址
    "OnvifPort": 80, // 端口
    "url": "rtsp://192.168.100.140:8554/3", // 视频地址
    "cap": "Content/BG/BG_5.jpg", // 截图地址
    "OnvifProfile": 0
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 3. 相机管理-获取视频地址(下拉框)

- **URL**: `/camera/alg_onvif_probe`
- **Method**: `GET`
- **Request Body**:
  ```json
  {
    "OnvifUser": "测试001", // 用户名
    "OnvifPassword": "123456", // 密码
    "DeviceIp": "0.0.0.0", // 网络地址
    "OnvifPort": 80, // 端口
    "OnvifProfile": 0
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "rtspId": 5,
          "rtspUrl": "rtsp://192.168.100.140:8554/3" // 地址
        }
      ],
      "message": "成功"
    }
    ```

### 4. 相机管理-新增相机

- **URL**: `/camera/app_rtsp_add`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "name": "测试", // 相机描述
    "url": "192.168.0.1", // 视频地址
    "cap": "" // 截图地址
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```

### 5. 相机管理-删除

- **URL**: `/camera/app_rtsp_remove`
- **Method**: `DELETE`
- **Request Body**:
  ```json
  {
    "id": 8 // id
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "删除成功"
    }
    ```
### 6. 识别管理-获取列表

- **URL**: `/polling/app_image_task_fetch`
- **Method**: `GET`
- **Request Body**:
  ```json
  {
    
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "algoId": 1001,
          "algoName": "清人清物算法",
          "rtspUrls": [
            {
              "rtspId": 5,
              "rtspName": "测试超载",
              "rtspUrl": "rtsp://192.168.100.140:8554/3"
            },
            {
              "rtspId": 6,
              "rtspName": "通道1 测试",
              "rtspUrl": "rtsp://192.168.100.140:8554/1"
            }
          ]
        }
      ],
      "message": "成功"
    }
    ```
### 7. 识别管理-保存

- **URL**: `/polling/app_image_task_config`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "algoId": 1001,
    "algoName": "清人清物算法",
    "RtspId": [
        5
    ]
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```
### 8. 巡视记录-获取列表

- **URL**: `/polling/app_ai_record`
- **Method**: `GET`
- **Request Body**:
  ```json
  {
   
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "data": [
        {
          "algoName": "车厢乘客密度-目标检测",                          // 执行算法
          "createTime": "1970-01-01 00:29:01",                        // 创建时间
          "finishedTime": "1970-01-01 00:29:07",                      // 完成时间
          "id": 35,
          "rtspNum": 2,                                               // 任务数量
          "status": 3,                                                // 运行状态
          "rtspUrls": [
              {
                  "message": "通过流地址获取图像失败",
                  "raw_path": null,
                  "rtspName": "测试超载",
                  "rtspUrl": "rtsp://192.168.100.140:8554/3",
                  "stage": 4
              },
              {
                  "message": "通过流地址获取图像失败",
                  "raw_path": null,
                  "rtspName": "通道1 测试",
                  "rtspUrl": "rtsp://192.168.100.140:8554/1",
                  "stage": 4
              }
          ],
        }
      ],
      "message": "成功"
    }
    ```
### 9. 巡视记录-获取清人清物底图 | 巡视-清人清物 | 巡视-乘客密度 | 巡视-乘客密度(基于目标检测)

- **URL**: `/camera/app_algorithm_start`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "algoId": 1001 | 1001 | 1002 | 1003,
    "type": 1 | 2 | 1 | 1,
  }
  ```
- **Response**:
  - **Status Code**: `200`
  - **Body**:
    ```json
    {
      "code": 200,
      "message": "成功"
    }
    ```
