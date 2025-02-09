choise = input("Вы хотите стать хостом(1) или подключиться(2): ")
if choise == "1":
    from translate_m2m import translate
    import rsa, threading, socket
else:
    import rsa, threading, socket

public_key, private_key = rsa.newkeys(2048)
print("generated")
public_partner = None

if choise == "1":
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("192.168.0.218", 9090))
    server.listen()
    client, _ = server.accept()
    client.send(public_key.save_pkcs1("PEM"))
    public_partner = rsa.PublicKey.load_pkcs1(client.recv(4096))
    print("CONNECTED")
elif choise == "2":
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.connect(("213.21.51.48", 9090))
    public_partner = rsa.PublicKey.load_pkcs1(client.recv(4096))
    client.send(public_key.save_pkcs1("PEM"))
    print("CONNECTED")
else:
    print("ERROR")
    exit()

def sending_message(c):
    while True:
        message = input("/")
        encrypted_message = rsa.encrypt(message.encode(),pub_key=public_partner)
        c.send(encrypted_message)
        # print("Вы: " + message)


def recv_message(c):
    while True:
        encrypted_message = c.recv(8192)
        decrypted_message = rsa.decrypt(encrypted_message,private_key).decode()
        print("Ответ: " + decrypted_message)
        if choise == "1":
            message = translate(decrypted_message)
            print("Текст: " + message[0])
            encrypted_message = rsa.encrypt(message[0].encode(), pub_key=public_partner)
            c.send(encrypted_message)




threading.Thread(target=sending_message,args=(client,)).start()
threading.Thread(target=recv_message,args=(client,)).start()