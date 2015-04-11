class CreateEmails < ActiveRecord::Migration
  def self.up
    create_table :emails do |t|
      t.column :id, :primary_key
      t.column :To, :string
      t.column :From, :string
      t.column :Date, :datetime
      t.column :Subject, :string
      t.column :Cc, :string
      t.column :Body, :text
      t.column :Hidden, :boolean, :default => false
      t.column :email_thread_id, :integer
      t.column :Submit_By, :string
      t.column :Submit_Date, :datetime
      t.column :Original, :text
    end
  end

  def self.down
    drop_table :emails
  end
end
